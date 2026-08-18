//! Authoritative nonblocking game state machine.

use crate::decision::{Budget, Failure, Reply};
use crate::error::SubmitError;
use crate::outcome::{DrawReason, MatchResult, NoContest, WinReason};
use hexo_engine::{Action, ActionId, Applied, Player, Position, TurnPhase};
use std::num::NonZeroU32;

/// What a driver-reported [`Failure`] costs the seat it happened to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FailurePolicy {
    /// The failing seat loses.
    #[default]
    Forfeit,
    /// The game is a no-contest.
    NoContest,
}

/// The match rules a game is played under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameSpec {
    /// Placements after which the game is [`DrawReason::PlyCap`], tested only on a
    /// placement that completed the mover's turn. Turns end at odd placement
    /// counts, so an even cap stops the game one placement past it rather than
    /// cutting a two-stone turn in half.
    pub ply_cap: NonZeroU32,
    /// What each seat is told it has to think with.
    pub budget: Budget,
    /// What a driver-reported failure costs.
    pub on_failure: FailurePolicy,
}

impl Default for GameSpec {
    fn default() -> Self {
        Self {
            ply_cap: NonZeroU32::new(512).unwrap(),
            budget: Budget::Unlimited,
            on_failure: FailurePolicy::Forfeit,
        }
    }
}

/// One placement as recorded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlyRecord {
    /// Who placed.
    pub seat: Player,
    /// What was placed, in the record encoding.
    pub action: ActionId,
    /// The position hash after the placement.
    pub zobrist_after: u64,
    /// Seat-owned bytes, stored verbatim and never interpreted.
    pub diagnostics: Option<Vec<u8>>,
}

/// What the game wants next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// A seat must choose.
    NeedDecision {
        /// Who must choose.
        seat: Player,
        /// The token this decision must be submitted with.
        generation: u64,
        /// What this seat is told it may spend.
        budget: Budget,
        /// The hash of the position being decided in.
        ///
        /// Mirror-keeping seats compare this with their own state. The seat,
        /// not the driver, authors [`crate::Decision::zobrist`].
        zobrist: u64,
        /// Placements made so far.
        ply: u32,
    },
    /// The game is over.
    Finished(MatchResult),
}

/// What an accepted submission did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transition {
    /// The placement, if the submission was one.
    pub applied: Option<Applied>,
    /// The position hash after the submission.
    pub zobrist: u64,
    /// The token the *next* decision must be submitted with.
    pub generation: u64,
    /// `Some` if the game just ended.
    pub result: Option<MatchResult>,
}

/// The one authoritative game.
#[derive(Clone, Debug)]
pub struct Game {
    spec: GameSpec,
    position: Position,
    generation: u64,
    result: Option<MatchResult>,
    plies: Vec<PlyRecord>,
}

impl Game {
    /// A new game under `spec`, at the empty position with `P0` to open.
    #[must_use]
    pub fn new(spec: GameSpec) -> Self {
        Self {
            spec,
            position: Position::new(),
            generation: 0,
            result: None,
            plies: Vec::new(),
        }
    }

    /// The match rules in force.
    #[inline]
    #[must_use]
    pub const fn spec(&self) -> &GameSpec {
        &self.spec
    }

    /// The canonical position, read-only.
    #[inline]
    #[must_use]
    pub const fn position(&self) -> &Position {
        &self.position
    }

    /// Every accepted placement so far, oldest first.
    #[inline]
    #[must_use]
    pub fn plies(&self) -> &[PlyRecord] {
        &self.plies
    }

    /// The result, if the game has ended.
    #[inline]
    #[must_use]
    pub const fn result(&self) -> Option<MatchResult> {
        self.result
    }

    /// The accepted move prefix, derived from [`Game::plies`].
    ///
    /// Replaying this prefix reproduces [`Game::position`].
    #[must_use]
    pub fn prefix(&self) -> Vec<Action> {
        self.plies
            .iter()
            .map(|ply| Action::from_id(ply.action))
            .collect()
    }

    /// What the game wants next.
    #[must_use]
    pub fn step(&self) -> Step {
        match self.result {
            Some(result) => Step::Finished(result),
            None => Step::NeedDecision {
                seat: self.position.current_player(),
                generation: self.generation,
                budget: self.spec.budget,
                zobrist: self.position.zobrist(),
                ply: self.position.stone_count(),
            },
        }
    }

    /// Report what a seat came back with.
    pub fn submit(&mut self, generation: u64, reply: Reply) -> Result<Transition, SubmitError> {
        if self.result.is_some() {
            return Err(SubmitError::Finished);
        }
        if generation != self.generation {
            return Err(SubmitError::StaleGeneration {
                expected: self.generation,
                got: generation,
            });
        }

        let seat = self.position.current_player();
        match reply {
            Reply::Resign => Ok(self.finish(MatchResult::Decisive {
                winner: seat.other(),
                reason: WinReason::Resignation,
            })),
            Reply::Failed(failure) => Ok(self.finish(match self.spec.on_failure {
                FailurePolicy::Forfeit => MatchResult::Decisive {
                    winner: seat.other(),
                    reason: match failure {
                        Failure::Timeout => WinReason::Timeout,
                        Failure::Crashed => WinReason::Crash,
                        Failure::Protocol => WinReason::Protocol,
                        Failure::Desync { expected, got } => WinReason::Desync { expected, got },
                    },
                },
                FailurePolicy::NoContest => {
                    MatchResult::NoContest(NoContest::SeatFailure { seat, failure })
                }
            })),
            Reply::Place(decision) => {
                let expected = self.position.zobrist();
                if decision.zobrist != expected {
                    return Err(SubmitError::Desync {
                        expected,
                        got: decision.zobrist,
                    });
                }
                match self.position.advance(decision.action) {
                    Ok(applied) => Ok(self.accept(seat, applied, decision.diagnostics)),
                    Err(error) if error.is_rule_violation() => {
                        Ok(self.finish(MatchResult::Decisive {
                            winner: seat.other(),
                            reason: WinReason::IllegalMove {
                                action: decision.action.id(),
                                cause: error,
                            },
                        }))
                    }
                    Err(error) => Ok(self.finish(MatchResult::NoContest(NoContest::EngineLimit {
                        seat,
                        error,
                    }))),
                }
            }
        }
    }

    /// Record an accepted placement and decide whether it ended the game.
    fn accept(
        &mut self,
        seat: Player,
        applied: Applied,
        diagnostics: Option<Vec<u8>>,
    ) -> Transition {
        let zobrist = self.position.zobrist();
        self.plies.push(PlyRecord {
            seat,
            action: applied.action.id(),
            zobrist_after: zobrist,
            diagnostics,
        });
        self.generation += 1;

        self.result = if let Some(outcome) = applied.outcome {
            Some(MatchResult::Decisive {
                winner: outcome.winner,
                reason: WinReason::SixInARow,
            })
        } else if matches!(applied.phase_after, TurnPhase::FirstStone)
            && self.position.stone_count() >= self.spec.ply_cap.get()
        {
            // `FirstStone` follows the opening and every completed two-stone turn,
            // so the cap is tested only at turn boundaries.
            Some(MatchResult::Drawn {
                reason: DrawReason::PlyCap,
            })
        } else {
            None
        };

        Transition {
            applied: Some(applied),
            zobrist,
            generation: self.generation,
            result: self.result,
        }
    }

    /// End the game without a placement.
    fn finish(&mut self, result: MatchResult) -> Transition {
        self.result = Some(result);
        Transition {
            applied: None,
            zobrist: self.position.zobrist(),
            generation: self.generation,
            result: Some(result),
        }
    }
}
