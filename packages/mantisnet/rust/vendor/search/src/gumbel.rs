//! Gumbel root sampling with sequential halving over deterministic lines.
//!
//! Each line starts from a sampled root action and is extended through the
//! evaluated position's prior argmax. Sequential halving selects which lines
//! receive additional depth.
//!
//! The root evaluation is outside the simulation budget. A round gives every
//! surviving line the same number of deepenings; any integer remainder is
//! unused. Terminal lines retain their exact root-frame value and emit no
//! evaluation request.

use crate::rng::SplitMix64;
use crate::seam::Evaluation;
use crate::session::{DecisionSession, LeafId, SessionStatus};
use hexo_engine::{Player, Position};
use hexo_runner::Decision;
use std::collections::{BTreeMap, VecDeque};
use std::num::{NonZeroU32, NonZeroUsize};

const C_VISIT: f64 = 50.0;
const C_SCALE: f64 = 1.0;
const UNIT_SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;

/// The explicit search shape for a [`GumbelSession`].
///
/// There is no `Default`; every field is required.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelConfig {
    /// Total equal-deepening allocations below the root.
    pub simulations: NonZeroU32,
    /// Maximum number of Gumbel-top root candidates.
    pub candidates: NonZeroUsize,
    /// Scale applied to every root Gumbel draw.
    ///
    /// Must be finite and non-negative. Zero removes root noise while retaining
    /// the searched line-value comparison; one leaves the draws unscaled.
    pub temperature: f64,
}

/// A completed search's canonical root-rank trace.
///
/// Candidate ranks are in Gumbel-top order. Each following row is the survivor
/// order after one completed halving round. Ranks always index the root
/// position's canonical legal order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GumbelTrace {
    candidate_root_ranks: Box<[usize]>,
    survivor_root_ranks: Box<[Box<[usize]>]>,
}

impl GumbelTrace {
    /// Candidate root ranks in initial Gumbel-top order.
    #[must_use]
    pub fn candidate_root_ranks(&self) -> &[usize] {
        &self.candidate_root_ranks
    }

    /// Survivor root ranks after each completed round.
    #[must_use]
    pub fn survivor_root_ranks(&self) -> &[Box<[usize]>] {
        &self.survivor_root_ranks
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Fresh,
    WantedRoot,
    AwaitingRoot(LeafId),
    Searching,
    AwaitingWave,
    Decided,
}

struct Line {
    root_rank: usize,
    root_log_prior: f64,
    gumbel: f64,
    position: Position,
    value: f64,
    policy_rank: Option<usize>,
    evaluated: bool,
    terminal: bool,
    visits: u32,
}

#[derive(Clone, Copy)]
struct Pending {
    line: usize,
    legal_count: usize,
}

/// A nonblocking Gumbel sequential-halving session.
///
/// `pump` emits one root question, then one equal-deepening wave at a time.
/// Answers inside a wave may be resumed in any order. Every emitted leaf has a
/// session-unique [`LeafId`], including across calls to
/// [`DecisionSession::begin`].
pub struct GumbelSession {
    config: GumbelConfig,
    rng: SplitMix64,
    injected: Option<VecDeque<f64>>,
    root: Position,
    root_player: Player,
    state: State,
    next_serial: u64,
    decision: Option<Decision>,
    lines: Vec<Line>,
    survivors: Vec<usize>,
    schedule: Vec<(usize, usize)>,
    round: usize,
    wave: usize,
    pending: BTreeMap<LeafId, Pending>,
    trace_candidates: Vec<usize>,
    trace_survivors: Vec<Box<[usize]>>,
    last_trace: Option<GumbelTrace>,
}

impl GumbelSession {
    /// Construct a session whose Gumbels come from its seeded [`SplitMix64`]
    /// stream.
    ///
    /// # Panics
    ///
    /// If `config.temperature` is not finite and non-negative.
    #[must_use]
    pub fn new(config: GumbelConfig, seed: u64) -> Self {
        Self::build(config, seed, None)
    }

    /// Construct a deterministic session whose Gumbels come only from
    /// `gumbels`.
    ///
    /// One root consumes one value per legal action, in canonical order. A
    /// positive budget below two consumes none because it falls back directly
    /// to prior argmax. Running out, or supplying a non-finite value, panics
    /// instead of silently switching to another random stream.
    ///
    /// # Panics
    ///
    /// If `config.temperature` is not finite and non-negative.
    #[must_use]
    pub fn with_gumbels(config: GumbelConfig, gumbels: impl IntoIterator<Item = f64>) -> Self {
        Self::build(config, 0, Some(gumbels.into_iter().collect()))
    }

    fn build(config: GumbelConfig, seed: u64, injected: Option<VecDeque<f64>>) -> Self {
        assert!(
            config.temperature.is_finite() && config.temperature >= 0.0,
            "GumbelConfig::temperature is {}; the root-noise scale must be finite and \
             non-negative",
            config.temperature,
        );
        Self {
            config,
            rng: SplitMix64::new(seed),
            injected,
            root: Position::new(),
            root_player: Player::P0,
            state: State::Fresh,
            next_serial: 0,
            decision: None,
            lines: Vec::new(),
            survivors: Vec::new(),
            schedule: Vec::new(),
            round: 0,
            wave: 0,
            pending: BTreeMap::new(),
            trace_candidates: Vec::new(),
            trace_survivors: Vec::new(),
            last_trace: None,
        }
    }

    /// Append deterministic Gumbels and make the session use only that queue.
    ///
    /// Once enabled, the injected queue remains authoritative across `reseed`
    /// and `begin`.
    pub fn queue_gumbels(&mut self, gumbels: impl IntoIterator<Item = f64>) {
        self.injected
            .get_or_insert_with(VecDeque::new)
            .extend(gumbels);
    }

    /// The most recently completed trace, if it has not been taken.
    #[must_use]
    pub fn last_trace(&self) -> Option<&GumbelTrace> {
        self.last_trace.as_ref()
    }

    /// Take the most recently completed trace.
    pub fn take_last_trace(&mut self) -> Option<GumbelTrace> {
        self.last_trace.take()
    }

    fn mint_leaf(&mut self) -> LeafId {
        let leaf = LeafId::from_serial(self.next_serial);
        self.next_serial = self
            .next_serial
            .checked_add(1)
            .expect("GumbelSession exhausted its leaf-id space");
        leaf
    }

    fn next_gumbel(&mut self) -> f64 {
        let value = if let Some(queue) = &mut self.injected {
            queue.pop_front().expect(
                "GumbelSession's injected-noise queue ran out; one finite Gumbel is required \
                 for every root legal action",
            )
        } else {
            // A half-unit offset turns SplitMix64's 53-bit mantissa into the
            // open interval (0, 1), where the inverse Gumbel CDF is finite.
            let unit = ((self.rng.next_u64() >> 11) as f64 + 0.5) * UNIT_SCALE;
            -(-unit.ln()).ln()
        };
        assert!(
            value.is_finite(),
            "GumbelSession received non-finite injected noise {value}"
        );
        value
    }

    fn accept_root(&mut self, evaluation: Evaluation) {
        let legal_count = self.root.legal_count();
        let max_candidates =
            usize::try_from(self.config.simulations.get() / 2).expect("half of a u32 fits usize");
        let candidate_count = self
            .config
            .candidates
            .get()
            .min(max_candidates)
            .min(legal_count);

        if candidate_count == 0 {
            let root_rank = prior_argmax(&evaluation.priors);
            self.author_decision(root_rank);
            return;
        }

        let mut scored = Vec::with_capacity(legal_count);
        for (root_rank, &prior) in evaluation.priors.iter().enumerate() {
            // Draw first and scale second so temperature never changes the RNG
            // stream, and T=1 retains the raw draw bit-for-bit.
            let gumbel = self.next_gumbel() * self.config.temperature;
            let log_prior = if prior == 0.0 {
                f64::NEG_INFINITY
            } else {
                f64::from(prior).ln()
            };
            scored.push((root_rank, gumbel, log_prior, gumbel + log_prior));
        }
        scored.sort_by(|a, b| b.3.total_cmp(&a.3).then_with(|| a.0.cmp(&b.0)));

        self.lines.clear();
        self.lines.reserve(candidate_count);
        for &(root_rank, gumbel, root_log_prior, _) in &scored[..candidate_count] {
            let mut position = self.root.clone();
            let action = position
                .nth_legal(root_rank)
                .expect("a rank selected from the root legal set");
            position
                .advance(action)
                .expect("a canonical root action is legal");
            let terminal = position.is_terminal();
            let value = if terminal {
                terminal_value(&position, self.root_player)
            } else {
                0.0
            };
            self.lines.push(Line {
                root_rank,
                root_log_prior,
                gumbel,
                position,
                value,
                policy_rank: None,
                evaluated: false,
                terminal,
                visits: 0,
            });
        }

        self.survivors.clear();
        self.survivors.extend(0..candidate_count);
        self.schedule = halving_schedule(self.config.simulations.get() as usize, candidate_count);
        self.round = 0;
        self.wave = 0;
        self.trace_candidates = self.lines.iter().map(|line| line.root_rank).collect();
        self.trace_survivors.clear();
        self.state = State::Searching;
    }

    fn pump_search(&mut self, emit: &mut dyn FnMut(LeafId, &Position)) -> SessionStatus {
        loop {
            if self.round == self.schedule.len() {
                self.finish_search();
                return SessionStatus::Decided;
            }

            let (scheduled_survivors, deepenings) = self.schedule[self.round];
            assert_eq!(
                scheduled_survivors,
                self.survivors.len(),
                "GumbelSession's halving schedule and survivor set diverged",
            );

            if self.wave == deepenings {
                self.finish_round();
                self.round += 1;
                self.wave = 0;
                continue;
            }

            self.pending.clear();
            let survivor_snapshot = self.survivors.clone();
            for line_index in survivor_snapshot {
                let line = &mut self.lines[line_index];
                if line.terminal {
                    continue;
                }

                if line.evaluated {
                    let policy_rank = line
                        .policy_rank
                        .expect("an evaluated line remembers its policy argmax");
                    let action = line
                        .position
                        .nth_legal(policy_rank)
                        .expect("the saved policy rank belongs to this line position");
                    line.position
                        .advance(action)
                        .expect("a canonical line action is legal");
                    if line.position.is_terminal() {
                        line.terminal = true;
                        line.value = terminal_value(&line.position, self.root_player);
                        line.visits += 1;
                        continue;
                    }
                }

                let legal_count = line.position.legal_count();
                let leaf = self.mint_leaf();
                let old = self.pending.insert(
                    leaf,
                    Pending {
                        line: line_index,
                        legal_count,
                    },
                );
                debug_assert!(old.is_none(), "leaf ids are unique");
                emit(leaf, &self.lines[line_index].position);
            }
            self.wave += 1;

            if self.pending.is_empty() {
                // Every surviving line was already terminal, or became
                // terminal on this extension. Continue synchronously.
                continue;
            }

            self.state = State::AwaitingWave;
            return SessionStatus::AwaitingEvals {
                in_flight: self.pending.len(),
            };
        }
    }

    fn finish_round(&mut self) {
        self.survivors
            .sort_by(|&a, &b| self.lines[b].value.total_cmp(&self.lines[a].value));
        self.survivors.truncate(self.survivors.len().div_ceil(2));
        self.trace_survivors.push(
            self.survivors
                .iter()
                .map(|&line| self.lines[line].root_rank)
                .collect(),
        );
    }

    fn finish_search(&mut self) {
        let max_visits = self
            .lines
            .iter()
            .map(|line| line.visits)
            .max()
            .expect("a searched root has a candidate");
        let mut best = self.survivors[0];
        let mut best_score = final_score(&self.lines[best], max_visits);
        for &candidate in &self.survivors[1..] {
            let score = final_score(&self.lines[candidate], max_visits);
            if score > best_score {
                best = candidate;
                best_score = score;
            }
        }
        self.author_decision(self.lines[best].root_rank);
    }

    fn author_decision(&mut self, root_rank: usize) {
        let action = self
            .root
            .nth_legal(root_rank)
            .expect("the selected rank belongs to the root legal set");
        self.decision = Some(Decision::new(action, self.root.zobrist()));
        self.last_trace = Some(GumbelTrace {
            candidate_root_ranks: self.trace_candidates.clone().into_boxed_slice(),
            survivor_root_ranks: self.trace_survivors.clone().into_boxed_slice(),
        });
        self.state = State::Decided;
    }
}

impl DecisionSession for GumbelSession {
    fn begin(&mut self, position: &Position) {
        assert!(
            !position.is_terminal(),
            "GumbelSession::begin on a terminal position; a driver only asks a live position's \
             mover",
        );
        self.root.clone_from(position);
        self.root_player = self.root.current_player();
        self.state = State::WantedRoot;
        self.decision = None;
        self.lines.clear();
        self.survivors.clear();
        self.schedule.clear();
        self.pending.clear();
        self.trace_candidates.clear();
        self.trace_survivors.clear();
    }

    fn pump(&mut self, emit: &mut dyn FnMut(LeafId, &Position)) -> SessionStatus {
        match self.state {
            State::Fresh => panic!("GumbelSession::pump before begin"),
            State::WantedRoot => {
                let leaf = self.mint_leaf();
                emit(leaf, &self.root);
                self.state = State::AwaitingRoot(leaf);
                SessionStatus::AwaitingEvals { in_flight: 1 }
            }
            State::AwaitingRoot(_) => SessionStatus::AwaitingEvals { in_flight: 1 },
            State::Searching => self.pump_search(emit),
            State::AwaitingWave => SessionStatus::AwaitingEvals {
                in_flight: self.pending.len(),
            },
            State::Decided => SessionStatus::Decided,
        }
    }

    fn resume(&mut self, leaf: LeafId, evaluation: Evaluation) {
        match self.state {
            State::AwaitingRoot(wanted) => {
                assert_eq!(
                    leaf, wanted,
                    "GumbelSession::resume with unknown {leaf:?}; the root leaf in flight is \
                     {wanted:?}",
                );
                evaluation.check(self.root.legal_count(), leaf);
                self.accept_root(evaluation);
            }
            State::AwaitingWave => {
                let pending = self.pending.remove(&leaf).unwrap_or_else(|| {
                    panic!(
                        "GumbelSession::resume with unknown {leaf:?}; {} line leaves are in flight",
                        self.pending.len(),
                    )
                });
                evaluation.check(pending.legal_count, leaf);
                let line = &mut self.lines[pending.line];
                line.policy_rank = Some(prior_argmax(&evaluation.priors));
                let sign = if line.position.current_player() == self.root_player {
                    1.0
                } else {
                    -1.0
                };
                line.value = sign * f64::from(evaluation.value);
                line.evaluated = true;
                line.visits += 1;
                if self.pending.is_empty() {
                    self.state = State::Searching;
                }
            }
            _ => panic!(
                "GumbelSession::resume with {leaf:?}, but the session has nothing matching in \
                 flight ({:?})",
                self.state,
            ),
        }
    }

    fn take_decision(&mut self) -> Option<Decision> {
        self.decision.take()
    }

    fn reseed(&mut self, seed: u64) {
        self.rng.reseed(seed);
    }
}

/// One live root candidate, as read by [`GumbelSession::live_snapshot`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineSnapshot {
    /// Index into the root position's canonical legal order.
    pub root_rank: usize,
    /// Deepenings this line has received so far.
    pub visits: u32,
    /// The line's current value in the root mover's frame.
    pub value: f64,
    /// The score sequential halving compares, at the snapshot's visit ceiling.
    pub score: f64,
    /// Whether the line has received its first evaluation (or is terminal).
    pub evaluated: bool,
    /// Whether the line reached a terminal position.
    pub terminal: bool,
}

/// Observational read of a running search, taken between pumps.
#[derive(Clone, Debug, PartialEq)]
pub struct GumbelSnapshot {
    /// Every root candidate, in Gumbel-top order.
    pub lines: Vec<LineSnapshot>,
    /// Indices into `lines` of the candidates still alive, in that same order.
    pub survivors: Vec<usize>,
    /// Completed halving rounds.
    pub round: usize,
    /// Total rounds in the schedule.
    pub rounds: usize,
    /// Deepenings delivered so far, summed over lines.
    pub completed_visits: u32,
    /// The configured simulation budget.
    pub target_visits: u32,
}

// Vendored addition (not in the upstream research crate): a pull-based,
// read-only snapshot for live search telemetry. The serve driver calls it
// between `pump` waves; nothing here mutates, allocates on the search path,
// draws randomness, or is reachable from the search itself, so a search runs
// bit-identically whether or not snapshots are ever taken.
impl GumbelSession {
    /// Snapshot the running (or just-decided) search, or `None` before the
    /// root evaluation has been accepted.
    #[must_use]
    pub fn live_snapshot(&self) -> Option<GumbelSnapshot> {
        if self.lines.is_empty() {
            return None;
        }
        let max_visits = self.lines.iter().map(|line| line.visits).max().unwrap_or(0);
        let lines = self
            .lines
            .iter()
            .map(|line| LineSnapshot {
                root_rank: line.root_rank,
                visits: line.visits,
                value: line.value,
                score: final_score(line, max_visits),
                evaluated: line.evaluated,
                terminal: line.terminal,
            })
            .collect();
        Some(GumbelSnapshot {
            lines,
            survivors: self.survivors.clone(),
            round: self.round,
            rounds: self.schedule.len(),
            completed_visits: self.lines.iter().map(|line| line.visits).sum(),
            target_visits: self.config.simulations.get(),
        })
    }
}

fn prior_argmax(priors: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..priors.len() {
        if priors[index] > priors[best] {
            best = index;
        }
    }
    best
}

fn terminal_value(position: &Position, root_player: Player) -> f64 {
    if position
        .outcome()
        .expect("terminal position has an outcome")
        .winner
        == root_player
    {
        1.0
    } else {
        -1.0
    }
}

fn final_score(line: &Line, max_visits: u32) -> f64 {
    line.gumbel + line.root_log_prior + (C_VISIT + f64::from(max_visits)) * C_SCALE * line.value
}

fn halving_schedule(simulations: usize, candidates: usize) -> Vec<(usize, usize)> {
    debug_assert!(candidates > 0);
    let rounds = (usize::BITS - (candidates - 1).leading_zeros()).max(1) as usize;
    let mut remaining = simulations;
    let mut survivors = candidates;
    let mut schedule = Vec::with_capacity(rounds);

    for round in 0..rounds {
        if remaining < survivors {
            break;
        }
        let share = remaining / survivors;
        let deepenings = if round + 1 == rounds {
            share
        } else {
            (1usize << round).min(share)
        };
        if deepenings == 0 {
            break;
        }
        schedule.push((survivors, deepenings));
        remaining -= survivors * deepenings;
        survivors = survivors.div_ceil(2);
    }
    schedule
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{Action, HexCoord};
    use hexo_runner::{Game, GameSpec, Reply, Step};

    fn game_after(moves: &[(i16, i16)]) -> Game {
        let mut game = Game::new(GameSpec::default());
        for &(q, r) in moves {
            let Step::NeedDecision { generation, .. } = game.step() else {
                panic!("fixture finished early")
            };
            let action = Action::new(HexCoord::new(q, r));
            let decision = Decision::new(action, game.position().zobrist());
            game.submit(generation, Reply::Place(decision))
                .expect("fixture move is legal");
        }
        game
    }

    fn config(simulations: u32, candidates: usize) -> GumbelConfig {
        GumbelConfig {
            simulations: NonZeroU32::new(simulations).expect("test budget is positive"),
            candidates: NonZeroUsize::new(candidates).expect("test width is positive"),
            temperature: 1.0,
        }
    }

    fn uniform(legal_count: usize, value: f32) -> Evaluation {
        Evaluation {
            priors: vec![1.0 / legal_count as f32; legal_count].into(),
            value,
        }
    }

    fn pump_leaves(session: &mut GumbelSession) -> Vec<(LeafId, usize, Player)> {
        let mut leaves = Vec::new();
        let status = session.pump(&mut |leaf, position| {
            leaves.push((leaf, position.legal_count(), position.current_player()));
        });
        match status {
            SessionStatus::AwaitingEvals { in_flight } => {
                assert_eq!(in_flight, leaves.len());
            }
            SessionStatus::Decided => assert!(leaves.is_empty()),
        }
        leaves
    }

    fn finish_uniform(session: &mut GumbelSession) -> usize {
        let mut evaluations = 0;
        loop {
            let leaves = pump_leaves(session);
            if leaves.is_empty() {
                return evaluations;
            }
            evaluations += leaves.len();
            for (leaf, legal_count, _) in leaves {
                session.resume(leaf, uniform(legal_count, 0.0));
            }
        }
    }

    fn root(session: &mut GumbelSession, game: &Game, priors: Vec<f32>) {
        session.begin(game.position());
        let leaves = pump_leaves(session);
        assert_eq!(leaves.len(), 1);
        session.resume(
            leaves[0].0,
            Evaluation {
                priors: priors.into(),
                value: 0.0,
            },
        );
    }

    #[test]
    fn schedule_matches_the_python_receipts() {
        assert_eq!(halving_schedule(16, 8), [(8, 1), (4, 2)]);
        assert_eq!(halving_schedule(16, 16), [(16, 1)]);
        assert_eq!(halving_schedule(32, 8), [(8, 1), (4, 2), (2, 8)]);
        assert_eq!(halving_schedule(32, 16), [(16, 1), (8, 2)]);
    }

    #[test]
    fn one_simulation_falls_back_to_stable_prior_argmax_without_noise() {
        let game = game_after(&[(0, 0)]);
        let n = game.position().legal_count();
        let mut priors = vec![0.0; n];
        priors[1] = 0.7;
        priors[2] = 0.7;
        let mut session = GumbelSession::with_gumbels(config(1, 16), []);

        root(&mut session, &game, priors);

        assert_eq!(
            session.take_decision().expect("fallback decides").action,
            game.position().nth_legal(1).expect("rank one"),
        );
        let trace = session.last_trace().expect("a completed trace exists");
        assert!(trace.candidate_root_ranks().is_empty());
        assert!(trace.survivor_root_ranks().is_empty());
    }

    #[test]
    fn zero_priors_have_negative_infinite_log_weight() {
        let game = game_after(&[(0, 0)]);
        let n = game.position().legal_count();
        let mut priors = vec![0.0; n];
        priors[1] = 1.0;
        let mut noise = vec![100.0; n];
        noise[1] = -100.0;
        let mut session = GumbelSession::with_gumbels(config(2, 1), noise);
        root(&mut session, &game, priors);

        assert_eq!(finish_uniform(&mut session), 2);
        assert_eq!(
            session.take_decision().expect("searched").action,
            game.position()
                .nth_legal(1)
                .expect("the sole positive prior"),
        );
        assert_eq!(
            session.last_trace().expect("trace").candidate_root_ranks(),
            [1],
        );
    }

    #[test]
    fn schedule_budget_and_survivor_trace_are_exact() {
        let game = game_after(&[(0, 0)]);
        let n = game.position().legal_count();
        let order = [3, 1, 7, 2, 6, 5, 4, 0];
        let mut noise = vec![-100.0; n];
        for (score, &rank) in order.iter().enumerate() {
            noise[rank] = 100.0 - score as f64;
        }
        let mut session = GumbelSession::with_gumbels(config(16, 8), noise);

        root(&mut session, &game, vec![1.0 / n as f32; n]);
        assert_eq!(
            finish_uniform(&mut session),
            16,
            "the root question is outside the simulation budget",
        );

        let trace = session.last_trace().expect("trace");
        assert_eq!(trace.candidate_root_ranks(), order);
        assert_eq!(
            trace.survivor_root_ranks(),
            [
                vec![3, 1, 7, 2].into_boxed_slice(),
                vec![3, 1].into_boxed_slice(),
            ],
        );
        let taken = session.take_last_trace().expect("take the trace once");
        assert_eq!(taken.candidate_root_ranks(), order);
        assert!(session.take_last_trace().is_none());
    }

    #[test]
    fn wave_answers_may_arrive_out_of_order_and_values_compare_movers() {
        let game = game_after(&[(0, 0)]);
        let root_player = game.position().current_player();
        let n = game.position().legal_count();
        let mut noise = vec![-100.0; n];
        noise[0] = 2.0;
        noise[1] = 1.0;
        let mut session = GumbelSession::with_gumbels(config(4, 2), noise);
        root(&mut session, &game, vec![1.0 / n as f32; n]);

        let first = pump_leaves(&mut session);
        assert_eq!(first.len(), 2);
        assert_ne!(first[0].0, first[1].0, "leaf ids are unique");
        assert_eq!(first[0].2, root_player);
        assert_eq!(first[1].2, root_player);
        session.resume(first[1].0, uniform(first[1].1, 0.2));
        session.resume(first[0].0, uniform(first[0].1, 0.8));

        let second = pump_leaves(&mut session);
        assert_eq!(second.len(), 2);
        assert_eq!(second[0].2, root_player.other());
        assert_eq!(second[1].2, root_player.other());
        // Own-frame +1 on line zero is root-frame -1. Own-frame -0.2 on
        // line one is root-frame +0.2, so line one must survive.
        session.resume(second[1].0, uniform(second[1].1, -0.2));
        session.resume(second[0].0, uniform(second[0].1, 1.0));

        assert!(pump_leaves(&mut session).is_empty());
        assert_eq!(
            session.take_decision().expect("searched").action,
            game.position().nth_legal(1).expect("safe root action"),
        );
        assert_eq!(
            session.last_trace().expect("trace").survivor_root_ranks(),
            [vec![1].into_boxed_slice()],
        );
    }

    #[test]
    fn an_immediate_terminal_is_frozen_without_a_leaf_question() {
        let moves = [
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (0, 3),
            (4, 0),
            (5, 0),
            (0, 5),
            (0, 7),
            (6, 0),
        ];
        let game = game_after(&moves);
        let winning = game
            .position()
            .legal_rank(Action::new(HexCoord::new(3, 0)))
            .expect("winning move is legal");
        let n = game.position().legal_count();
        let mut priors = vec![0.0; n];
        priors[winning] = 1.0;
        let mut session = GumbelSession::with_gumbels(config(2, 1), vec![0.0; n]);

        root(&mut session, &game, priors);

        assert!(pump_leaves(&mut session).is_empty());
        assert_eq!(
            session
                .take_decision()
                .expect("terminal line decides")
                .action,
            game.position().nth_legal(winning).expect("winning action"),
        );
        let trace = session.last_trace().expect("trace");
        assert_eq!(trace.candidate_root_ranks(), [winning]);
        assert_eq!(
            trace.survivor_root_ranks(),
            [vec![winning].into_boxed_slice()],
        );
    }

    #[test]
    #[should_panic(expected = "unknown")]
    fn a_stale_leaf_from_an_abandoned_begin_is_refused() {
        let game = game_after(&[(0, 0)]);
        let mut session = GumbelSession::new(config(2, 1), 7);
        session.begin(game.position());
        let stale = pump_leaves(&mut session)[0].0;
        session.begin(game.position());
        let fresh = pump_leaves(&mut session)[0].0;
        assert_ne!(stale, fresh);
        session.resume(stale, uniform(game.position().legal_count(), 0.0));
    }

    #[test]
    #[should_panic(expected = "priors but the evaluated position has")]
    fn a_root_answer_with_the_wrong_action_count_is_refused() {
        let game = game_after(&[(0, 0)]);
        let mut session = GumbelSession::new(config(2, 1), 7);
        session.begin(game.position());
        let root_leaf = pump_leaves(&mut session)[0].0;
        session.resume(root_leaf, uniform(2, 0.0));
    }

    #[test]
    #[should_panic(expected = "injected-noise queue ran out")]
    fn deterministic_noise_never_falls_back_when_its_queue_runs_out() {
        let game = game_after(&[(0, 0)]);
        let n = game.position().legal_count();
        let mut session = GumbelSession::with_gumbels(config(2, 1), [0.0]);
        root(&mut session, &game, vec![1.0 / n as f32; n]);
    }

    #[test]
    #[should_panic(expected = "non-finite injected noise")]
    fn non_finite_injected_noise_is_refused() {
        let game = Game::new(GameSpec::default());
        let mut session = GumbelSession::with_gumbels(config(2, 1), [f64::NAN]);
        root(&mut session, &game, vec![1.0]);
    }

    #[test]
    #[should_panic(expected = "pump before begin")]
    fn pumping_before_begin_is_refused() {
        GumbelSession::new(config(2, 1), 7).pump(&mut |_leaf, _position| {});
    }
}
