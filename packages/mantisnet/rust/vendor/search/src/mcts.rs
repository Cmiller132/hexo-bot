//! PUCT with virtual loss and an in-flight cap.

use crate::rng::SplitMix64;
use crate::seam::Evaluation;
use crate::select::{Child, SearchOutcome, SelectFromSearch};
use crate::session::{DecisionSession, LeafId, SessionStatus};
use hexo_engine::{Action, Player, Position, Search};
use hexo_runner::Decision;
use std::num::{NonZeroU32, NonZeroUsize};

/// The shape of one tree search.
///
/// Every field is required; this type has no `Default`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsConfig {
    /// Visits below the root, and the whole compute budget of one decision.
    ///
    /// A *visit* is one descent from the root: it ends either at a leaf whose
    /// evaluation is requested or at a placement that wins on the spot, and
    /// either way it lands one visit on exactly one root child. The root's own
    /// evaluation — the one that supplies its priors — is not a visit, so a
    /// budget of `n` dispatches `n` descents and at most `n + 1` evaluations.
    pub visits: NonZeroU32,
    /// How many leaves this session may have out for evaluation at once.
    ///
    /// The session returns after reaching this cap until results are resumed.
    pub max_in_flight: NonZeroUsize,
    /// The PUCT exploration constant. Must be finite and non-negative; zero is
    /// meaningful (pure exploitation of the value estimate), so it is not a
    /// `NonZero`.
    pub c_puct: f32,
}

/// Index into [`Tree::nodes`]; a decision's tree is bounded by its visit budget.
type NodeIx = u32;

/// An edge whose child node has not been created yet.
const NO_NODE: NodeIx = u32::MAX;

/// The root of every tree.
const ROOT: NodeIx = 0;

/// One action out of a node, with the statistics kept from its parent's side.
#[derive(Clone, Copy, Debug)]
struct Edge {
    action: Action,
    /// The parent's prior for this action, from the parent's evaluation.
    prior: f32,
    /// The node this edge leads to, or [`NO_NODE`] until a descent creates it.
    child: NodeIx,
    /// Real visits plus outstanding virtual losses.
    visits: u32,
    /// Sum of backed-up values in the **parent's** perspective.
    total_value: f64,
}

/// One position in the tree.
#[derive(Clone, Copy, Debug)]
struct Node {
    /// The player to move at this node's position.
    ///
    /// Backpropagation compares movers because consecutive plies can have the
    /// same mover.
    mover: Player,
    /// Start of this node's edges in [`Tree::edges`]; the children of one node
    /// are contiguous.
    first_edge: u32,
    /// How many edges; zero until the node is first emitted as a leaf.
    edge_count: u32,
    /// Whether this node's evaluation has arrived. Until it has, the node is a
    /// leaf: its edges carry no priors and nothing selects through it.
    evaluated: bool,
    /// Sum of this node's edge visits, the `N_parent` term in PUCT.
    visits: u32,
}

/// One selection-path step: parent node and selected edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Step {
    node: NodeIx,
    edge: u32,
}

/// The tree arena. Cleared between decisions, never deallocated.
#[derive(Debug, Default)]
struct Tree {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl Tree {
    /// Drop the tree, keeping both arenas.
    fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
    }

    /// Add an unevaluated, childless node whose position has `mover` to move.
    fn push_node(&mut self, mover: Player) -> NodeIx {
        let ix = NodeIx::try_from(self.nodes.len()).expect("a tree is bounded by its visit budget");
        self.nodes.push(Node {
            mover,
            first_edge: 0,
            edge_count: 0,
            evaluated: false,
            visits: 0,
        });
        ix
    }

    /// This node's edges, in canonical legal order.
    fn edges_of(&self, node: NodeIx) -> &[Edge] {
        let n = self.nodes[node as usize];
        &self.edges[n.first_edge as usize..][..n.edge_count as usize]
    }

    /// Materialise `node`'s children as `position`'s legal actions in canonical
    /// order, with no priors yet.
    fn create_edges(&mut self, node: NodeIx, position: &Position) {
        debug_assert_eq!(self.nodes[node as usize].edge_count, 0);
        let first = u32::try_from(self.edges.len()).expect("the edge arena fits u32");
        self.edges
            .extend(position.legal_actions().map(|action| Edge {
                action,
                prior: 0.0,
                child: NO_NODE,
                visits: 0,
                total_value: 0.0,
            }));
        let end = u32::try_from(self.edges.len()).expect("the edge arena fits u32");
        let n = &mut self.nodes[node as usize];
        n.first_edge = first;
        n.edge_count = end - first;
        debug_assert_eq!(n.edge_count as usize, position.legal_count());
    }

    /// Write the evaluation's priors onto `node`'s edges. Index `i` of the
    /// priors is edge `i`, which is `nth_legal(i)` of the node's position —
    /// the one convention every model package is held to.
    fn set_priors(&mut self, node: NodeIx, priors: &[f32]) {
        let n = self.nodes[node as usize];
        debug_assert_eq!(n.edge_count as usize, priors.len());
        for (edge, &prior) in self.edges[n.first_edge as usize..][..n.edge_count as usize]
            .iter_mut()
            .zip(priors)
        {
            edge.prior = prior;
        }
    }

    /// Charge every edge of `path` one visit and one lost unit, so a concurrent
    /// descent sees this branch as losing and goes somewhere else.
    fn apply_virtual_loss(&mut self, path: &[Step]) {
        for step in path {
            let edge = &mut self.edges[step.edge as usize];
            edge.visits += 1;
            edge.total_value -= 1.0;
            self.nodes[step.node as usize].visits += 1;
        }
    }

    /// The exact inverse of [`Tree::apply_virtual_loss`], applied before the real
    /// value so that the two are never entangled.
    fn undo_virtual_loss(&mut self, path: &[Step]) {
        for step in path {
            let edge = &mut self.edges[step.edge as usize];
            debug_assert!(edge.visits > 0, "a virtual loss was removed twice");
            edge.visits -= 1;
            edge.total_value += 1.0;
            self.nodes[step.node as usize].visits -= 1;
        }
    }

    /// Back up `value`, stated from `perspective`'s side, along `path`.
    ///
    /// Each edge stores values in its parent's perspective, so sign is selected
    /// by comparing the parent mover with `perspective`.
    fn credit(&mut self, path: &[Step], perspective: Player, value: f64) {
        for step in path {
            let signed = if self.nodes[step.node as usize].mover == perspective {
                value
            } else {
                -value
            };
            let edge = &mut self.edges[step.edge as usize];
            edge.visits += 1;
            edge.total_value += signed;
            self.nodes[step.node as usize].visits += 1;
        }
    }
}

/// One leaf whose evaluation has been asked for and not yet answered.
#[derive(Debug)]
struct InFlight {
    leaf: LeafId,
    node: NodeIx,
    /// The descent path used to remove virtual loss and credit the result.
    path: Vec<Step>,
}

/// The tree and the bookkeeping that walks it.
///
/// Split out from [`MctsSession`] so that the `Search` holding `&mut` on the
/// session's root position and the code mutating the tree are borrows of two
/// different fields.
#[derive(Debug, Default)]
struct Walker {
    tree: Tree,
    in_flight: Vec<InFlight>,
    /// Path buffers handed back by resumed leaves.
    path_pool: Vec<Vec<Step>>,
    /// The current descent's path. Owned by the walker so a descent allocates
    /// nothing.
    path: Vec<Step>,
    /// Visits taken from the budget so far.
    dispatched: u32,
    /// Monotonic leaf serial that is not reset between decisions.
    next_serial: u64,
}

impl Walker {
    /// Start a fresh tree rooted at a position with `mover` to move.
    fn restart(&mut self, mover: Player) {
        self.tree.clear();
        for entry in self.in_flight.drain(..) {
            self.path_pool.push(entry.path);
        }
        self.dispatched = 0;
        self.path.clear();
        let root = self.tree.push_node(mover);
        debug_assert_eq!(root, ROOT);
    }

    /// The edge of `node` with the highest PUCT score.
    ///
    /// Uses `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`, with `Q = 0`
    /// for an unvisited child.
    ///
    /// At `N_parent == 0`, all scores are zero and canonical order breaks the
    /// tie.
    fn select_edge(&self, node: NodeIx, c_puct: f64) -> u32 {
        let n = self.tree.nodes[node as usize];
        debug_assert!(n.edge_count > 0, "selecting from a node with no children");
        let parent = f64::from(n.visits).sqrt();
        let mut best = n.first_edge;
        let mut best_score = f64::NEG_INFINITY;
        for ix in n.first_edge..n.first_edge + n.edge_count {
            let edge = self.tree.edges[ix as usize];
            let q = if edge.visits == 0 {
                0.0
            } else {
                edge.total_value / f64::from(edge.visits)
            };
            let score =
                q + c_puct * f64::from(edge.prior) * parent / (1.0 + f64::from(edge.visits));
            if score > best_score {
                best_score = score;
                best = ix;
            }
        }
        best
    }

    /// One descent from the root, ending at a leaf whose evaluation is requested
    /// or at a placement that wins on the spot. Always returns to the root.
    fn descend(
        &mut self,
        search: &mut Search<'_>,
        c_puct: f64,
        emit: &mut dyn FnMut(LeafId, &Position),
    ) {
        debug_assert!(search.at_floor(), "a descent starts from the root position");
        self.path.clear();
        let mut node = ROOT;
        loop {
            if !self.tree.nodes[node as usize].evaluated {
                self.emit_leaf(node, search.position(), emit);
                break;
            }
            let edge = self.select_edge(node, c_puct);
            self.path.push(Step { node, edge });
            let action = self.tree.edges[edge as usize].action;
            let applied = search
                .apply(action)
                .expect("a tree edge was legal at its parent, and the descent rebuilt that parent");
            if let Some(outcome) = applied.outcome {
                // Terminal outcomes consume a visit without an evaluation.
                self.tree.credit(&self.path, outcome.winner, 1.0);
                self.dispatched += 1;
                break;
            }
            let child = self.tree.edges[edge as usize].child;
            node = if child == NO_NODE {
                let fresh = self.tree.push_node(search.position().current_player());
                self.tree.edges[edge as usize].child = fresh;
                fresh
            } else {
                child
            };
        }
        // Unwind here because this `Search` is reused by the next descent.
        search.unwind();
        debug_assert!(search.at_floor(), "a descent returns to the root position");
    }

    /// Ask for `node`'s evaluation, charge the path a virtual loss, and take a
    /// visit from the budget unless this is the root's own evaluation.
    fn emit_leaf(
        &mut self,
        node: NodeIx,
        position: &Position,
        emit: &mut dyn FnMut(LeafId, &Position),
    ) {
        debug_assert!(
            !position.is_terminal(),
            "a terminal position is backed up on the spot and never becomes a node",
        );
        let leaf = LeafId::from_serial(self.next_serial);
        self.next_serial += 1;
        emit(leaf, position);

        // Materialize children while the transient position is available.
        // Priors remain unreadable until `evaluated` becomes true.
        //
        // A node still in flight may be emitted more than once; only the first
        // emission builds its edges.
        if self.tree.nodes[node as usize].edge_count == 0 {
            self.tree.create_edges(node, position);
        }

        let mut path = self.path_pool.pop().unwrap_or_default();
        path.clear();
        path.extend_from_slice(&self.path);
        self.in_flight.push(InFlight { leaf, node, path });
        self.tree.apply_virtual_loss(&self.path);

        // The root's own evaluation is not a visit: a visit is a descent, and a
        // descent is what lands a visit on a root child. See `MctsConfig::visits`.
        if !self.path.is_empty() {
            self.dispatched += 1;
        }
    }

    /// Hand a resumed leaf's path buffer back to the pool.
    fn recycle(&mut self, path: Vec<Step>) {
        self.path_pool.push(path);
    }

    /// Assert root visit accounting, including outstanding virtual losses.
    ///
    /// Root-edge visits and the maintained root total must both equal
    /// `dispatched` before, during, and after resumes.
    fn debug_assert_accounting(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        if !self.tree.nodes[ROOT as usize].evaluated {
            return;
        }
        let charged: u32 = self.tree.edges_of(ROOT).iter().map(|e| e.visits).sum();
        debug_assert_eq!(
            charged, self.tree.nodes[ROOT as usize].visits,
            "the root's maintained visit total drifted from its edges",
        );
        let in_flight = self.in_flight.iter().filter(|f| !f.path.is_empty()).count();
        debug_assert_eq!(
            charged as usize, self.dispatched as usize,
            "root-child visits ({charged}), of which {in_flight} are still in flight, must equal \
             the visits dispatched ({})",
            self.dispatched,
        );
        debug_assert!(
            in_flight <= self.dispatched as usize,
            "more leaves in flight than visits dispatched",
        );
    }
}

/// A seat that searches with PUCT, batching its leaf evaluations instead of
/// blocking on them.
///
/// [`DecisionSession::begin`] clears the tree while retaining arena capacity.
/// Sessions do not reuse subtrees or maintain a transposition table.
pub struct MctsSession {
    config: MctsConfig,
    selector: Box<dyn SelectFromSearch>,
    rng: SplitMix64,
    /// The session's own copy of the game's position, and the floor every
    /// descent unwinds back to.
    root: Position,
    walker: Walker,
    /// The root children handed to the selector, rebuilt once per decision.
    summary: Vec<Child>,
    begun: bool,
    authored: bool,
    decision: Option<Decision>,
}

impl MctsSession {
    /// A session searching under `config`, selecting with `selector`, sampling
    /// from `seed`.
    ///
    /// # Panics
    ///
    /// If `config.c_puct` is not finite and non-negative.
    #[must_use]
    pub fn new(config: MctsConfig, selector: Box<dyn SelectFromSearch>, seed: u64) -> Self {
        assert!(
            config.c_puct.is_finite() && config.c_puct >= 0.0,
            "MctsConfig::c_puct is {}; the exploration constant must be finite and non-negative",
            config.c_puct,
        );
        Self {
            config,
            selector,
            rng: SplitMix64::new(seed),
            root: Position::new(),
            walker: Walker::default(),
            summary: Vec::new(),
            begun: false,
            authored: false,
            decision: None,
        }
    }

    /// The search shape in force.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> MctsConfig {
        self.config
    }

    /// Author the decision if the budget is spent and nothing is outstanding.
    fn finish_if_ready(&mut self) {
        if self.authored
            || self.walker.dispatched < self.config.visits.get()
            || !self.walker.in_flight.is_empty()
        {
            return;
        }

        let Self {
            root,
            walker,
            summary,
            selector,
            rng,
            decision,
            authored,
            ..
        } = self;

        summary.clear();
        summary.extend(walker.tree.edges_of(ROOT).iter().map(|edge| Child {
            action: edge.action,
            visits: edge.visits,
            mean_value: if edge.visits == 0 {
                0.0
            } else {
                edge.total_value / f64::from(edge.visits)
            },
            prior: edge.prior,
        }));

        let outcome = SearchOutcome::new(root, summary);
        let action = selector.select(&outcome, rng);
        let diagnostics = selector.diagnostics(&outcome);
        // The decision attests the searched root and preserves package
        // diagnostics. The game owns placement adjudication.
        *decision = Some(Decision {
            action,
            zobrist: root.zobrist(),
            diagnostics,
        });
        *authored = true;
    }
}

impl DecisionSession for MctsSession {
    fn begin(&mut self, position: &Position) {
        assert!(
            !position.is_terminal(),
            "MctsSession::begin on a terminal position; a driver only asks a live position's \
             mover",
        );
        self.root.clone_from(position);
        self.walker.restart(self.root.current_player());
        self.summary.clear();
        self.begun = true;
        self.authored = false;
        self.decision = None;
    }

    fn pump(&mut self, emit: &mut dyn FnMut(LeafId, &Position)) -> SessionStatus {
        assert!(self.begun, "MctsSession::pump before begin");
        if self.authored {
            return SessionStatus::Decided;
        }

        if !self.walker.tree.nodes[ROOT as usize].evaluated {
            // Nothing below the root can be selected until the root's own
            // evaluation supplies its priors, so this is the whole of this pump.
            debug_assert!(self.walker.in_flight.len() <= 1);
            if self.walker.in_flight.is_empty() {
                self.walker.path.clear();
                self.walker.emit_leaf(ROOT, &self.root, emit);
            }
            return SessionStatus::AwaitingEvals {
                in_flight: self.walker.in_flight.len(),
            };
        }

        let c_puct = f64::from(self.config.c_puct);
        let budget = self.config.visits.get();
        let cap = self.config.max_in_flight.get();
        {
            let Self { root, walker, .. } = self;
            let mut search = Search::new(root);
            while walker.dispatched < budget && walker.in_flight.len() < cap {
                walker.descend(&mut search, c_puct, emit);
            }
        }
        self.walker.debug_assert_accounting();

        self.finish_if_ready();
        if self.authored {
            SessionStatus::Decided
        } else {
            SessionStatus::AwaitingEvals {
                in_flight: self.walker.in_flight.len(),
            }
        }
    }

    fn resume(&mut self, leaf: LeafId, evaluation: Evaluation) {
        assert!(self.begun, "MctsSession::resume before begin");
        let walker = &mut self.walker;
        let slot = walker
            .in_flight
            .iter()
            .position(|entry| entry.leaf == leaf)
            .unwrap_or_else(|| {
                let outstanding: Vec<LeafId> =
                    walker.in_flight.iter().map(|entry| entry.leaf).collect();
                panic!(
                    "MctsSession::resume with unknown {leaf:?}; this session is waiting on \
                     {outstanding:?}. A leaf id is never reused, so an id it does not recognise \
                     is an answer to a question some other session asked, or to one this session \
                     asked before its last begin",
                )
            });

        // Validate before mutating in-flight state.
        let node = walker.tree.nodes[walker.in_flight[slot].node as usize];
        evaluation.check(node.edge_count as usize, leaf);

        let entry = walker.in_flight.swap_remove(slot);
        walker.tree.undo_virtual_loss(&entry.path);
        if !node.evaluated {
            walker.tree.set_priors(entry.node, &evaluation.priors);
            walker.tree.nodes[entry.node as usize].evaluated = true;
        }
        // The value is stated from the leaf's side to move, which is this node's
        // recorded mover.
        walker
            .tree
            .credit(&entry.path, node.mover, f64::from(evaluation.value));
        walker.recycle(entry.path);
        walker.debug_assert_accounting();

        self.finish_if_ready();
    }

    fn take_decision(&mut self) -> Option<Decision> {
        self.decision.take()
    }

    fn reseed(&mut self, seed: u64) {
        self.rng.reseed(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{Action, HexCoord};
    use hexo_runner::{Game, GameSpec, Reply, Step as GameStep};

    /// A selector that always plays the most-visited child.
    struct MaxVisits;

    impl SelectFromSearch for MaxVisits {
        fn select(&mut self, outcome: &SearchOutcome<'_>, _rng: &mut SplitMix64) -> Action {
            outcome
                .children()
                .iter()
                .max_by_key(|c| c.visits)
                .expect("a live root has children")
                .action
        }

        fn diagnostics(&mut self, _outcome: &SearchOutcome<'_>) -> Option<Vec<u8>> {
            None
        }
    }

    fn config(visits: u32, cap: usize) -> MctsConfig {
        MctsConfig {
            visits: NonZeroU32::new(visits).expect("nonzero"),
            max_in_flight: NonZeroUsize::new(cap).expect("nonzero"),
            c_puct: 1.5,
        }
    }

    fn opened_game() -> Game {
        let mut game = Game::new(GameSpec::default());
        let GameStep::NeedDecision { generation, .. } = game.step() else {
            unreachable!("a new game wants a decision")
        };
        let zobrist = game.position().zobrist();
        game.submit(
            generation,
            Reply::Place(Decision::new(Action::new(HexCoord::ORIGIN), zobrist)),
        )
        .expect("the opening is legal");
        game
    }

    /// Drive one decision with uniform priors and value zero.
    fn run(session: &mut MctsSession, game: &Game) -> Decision {
        session.begin(game.position());
        let mut leaves: Vec<(LeafId, usize)> = Vec::new();
        loop {
            leaves.clear();
            let status = session.pump(&mut |leaf, position| {
                leaves.push((leaf, position.legal_count()));
            });
            if status == SessionStatus::Decided {
                break;
            }
            assert!(!leaves.is_empty(), "an awaiting pump that emitted nothing");
            for (leaf, n) in leaves.drain(..) {
                let priors = vec![1.0 / n as f32; n].into_boxed_slice();
                session.resume(leaf, Evaluation { priors, value: 0.0 });
            }
        }
        session.take_decision().expect("decided")
    }

    #[test]
    fn beginning_again_reuses_the_tree_arenas() {
        let game = opened_game();
        let mut session = MctsSession::new(config(8, 2), Box::new(MaxVisits), 1);
        run(&mut session, &game);

        let nodes = session.walker.tree.nodes.capacity();
        let edges = session.walker.tree.edges.capacity();
        assert!(nodes > 0 && edges > 0);

        session.begin(game.position());
        assert_eq!(session.walker.tree.nodes.len(), 1, "only the root survives");
        assert!(session.walker.tree.edges.is_empty());
        assert_eq!(session.walker.tree.nodes.capacity(), nodes);
        assert_eq!(session.walker.tree.edges.capacity(), edges);
    }

    #[test]
    fn resumed_leaves_return_their_path_buffers() {
        let game = opened_game();
        let mut session = MctsSession::new(config(16, 4), Box::new(MaxVisits), 1);
        run(&mut session, &game);
        assert!(
            !session.walker.path_pool.is_empty(),
            "every resumed leaf hands its path buffer back",
        );
        assert!(session.walker.in_flight.is_empty());
    }

    #[test]
    fn the_budget_counts_descents_and_not_the_root_evaluation() {
        let game = opened_game();
        let mut session = MctsSession::new(config(12, 3), Box::new(MaxVisits), 1);
        run(&mut session, &game);
        assert_eq!(session.walker.dispatched, 12);
        let visits: u32 = session
            .walker
            .tree
            .edges_of(ROOT)
            .iter()
            .map(|e| e.visits)
            .sum();
        assert_eq!(visits, 12, "every visit landed on a root child");
    }

    #[test]
    fn leaf_ids_are_never_reused_across_decisions() {
        let game = opened_game();
        let mut session = MctsSession::new(config(4, 1), Box::new(MaxVisits), 1);
        let mut seen = Vec::new();
        for _ in 0..3 {
            session.begin(game.position());
            loop {
                let mut round = Vec::new();
                let status = session.pump(&mut |leaf, position| {
                    round.push((leaf, position.legal_count()));
                });
                if status == SessionStatus::Decided {
                    break;
                }
                for (leaf, n) in round {
                    seen.push(leaf);
                    let priors = vec![1.0 / n as f32; n].into_boxed_slice();
                    session.resume(leaf, Evaluation { priors, value: 0.0 });
                }
            }
            session.take_decision().expect("decided");
        }
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), seen.len(), "a leaf id was minted twice");
    }
}
