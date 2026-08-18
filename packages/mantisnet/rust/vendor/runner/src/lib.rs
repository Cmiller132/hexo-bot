//! Hexo match state, submissions, and adjudication.
//!
//! ```
//! use hexo_runner::{Decision, Game, GameSpec, Reply, Step};
//!
//! let mut game = Game::new(GameSpec::default());
//! loop {
//!     match game.step() {
//!         Step::Finished(result) => break result,
//!         Step::NeedDecision { generation, zobrist, .. } => {
//!             // The driver obtains a seat decision outside the game state machine.
//!             let action = game.position().nth_legal(0).expect("a legal move");
//!             let reply = Reply::Place(Decision::new(action, zobrist));
//!             game.submit(generation, reply).expect("fresh generation");
//!         }
//!     }
//! };
//! ```

pub mod decision;
pub mod error;
pub mod game;
pub mod outcome;

pub use decision::{Budget, Decision, Failure, Reply};
pub use error::SubmitError;
pub use game::{FailurePolicy, Game, GameSpec, PlyRecord, Step, Transition};
pub use outcome::{DrawReason, MatchResult, NoContest, WinReason};

/// Version of the runner decision/result model and native seat message set.
pub const PROTOCOL_VERSION: u32 = 2;
