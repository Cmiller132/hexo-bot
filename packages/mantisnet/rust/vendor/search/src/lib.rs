//! Evaluator batching and nonblocking decision sessions.
//!
//! A model package supplies the [`Encoder`], [`Evaluator`], and selection
//! policy. This crate owns the session state machines and their batching seam.
//!
//! [`PolicySession`] evaluates one root per move. [`MctsSession`] implements PUCT
//! with virtual loss and an in-flight cap. [`GumbelSession`] applies sequential
//! halving to sampled root lines. All three implement [`DecisionSession`].
//!
//! ```
//! use hexo_engine::{Action, Position};
//! use hexo_runner::{Game, GameSpec, Reply, Step};
//! use hexo_search::{
//!     DecisionSession, EncodedBatch, Encoder, Evaluation, Evaluator, PolicySession,
//!     SelectFromPolicy, SessionStatus, SplitMix64,
//! };
//! use std::num::NonZeroU32;
//!
//! // The encoder, evaluator, and selector are package-owned.
//!
//! /// Encodes the legal-action count.
//! struct Count;
//! impl Encoder for Count {
//!     fn encode(&self, position: &Position, out: &mut Vec<u8>) {
//!         out.extend_from_slice(&(position.legal_count() as u32).to_le_bytes());
//!     }
//! }
//!
//! /// Returns a uniform evaluation for each batch item.
//! struct Uniform;
//! impl Evaluator for Uniform {
//!     fn evaluate(&mut self, batch: &EncodedBatch, out: &mut Vec<Evaluation>) {
//!         for item in batch.iter() {
//!             let n = u32::from_le_bytes(item.try_into().expect("four bytes")) as usize;
//!             out.push(Evaluation {
//!                 priors: vec![1.0 / n as f32; n].into(),
//!                 value: 0.0,
//!             });
//!         }
//!     }
//! }
//!
//! /// Selects the highest-prior action.
//! struct Highest;
//! impl SelectFromPolicy for Highest {
//!     fn select(&mut self, root: &Position, e: &Evaluation, _rng: &mut SplitMix64) -> Action {
//!         let best = e.priors.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1));
//!         let (index, _) = best.expect("a live position has a legal action");
//!         root.nth_legal(index).expect("an index into the canonical legal order")
//!     }
//!     fn diagnostics(&mut self, _root: &Position, _e: &Evaluation) -> Option<Vec<u8>> {
//!         None
//!     }
//! }
//!
//! let spec = GameSpec { ply_cap: NonZeroU32::new(6).expect("nonzero"), ..GameSpec::default() };
//! let mut game = Game::new(spec);
//! let mut seats = [
//!     PolicySession::new(Box::new(Highest), 0x5eed),
//!     PolicySession::new(Box::new(Highest), 0xd33d),
//! ];
//! let mut evaluator = Uniform;
//! let mut batch = EncodedBatch::new();
//! let mut leaves = Vec::new();
//! let mut answers = Vec::new();
//!
//! let result = loop {
//!     let Step::NeedDecision { seat, generation, .. } = game.step() else {
//!         break game.result().expect("a finished game has a result");
//!     };
//!     let session = &mut seats[seat.index()];
//!     session.begin(game.position());
//!     loop {
//!         batch.clear();
//!         leaves.clear();
//!         // Emitted leaves are encoded before their transient positions expire.
//!         let status = session.pump(&mut |leaf, position| {
//!             leaves.push(leaf);
//!             batch.push_with(&Count, position);
//!         });
//!         if status == SessionStatus::Decided {
//!             break;
//!         }
//!         answers.clear();
//!         evaluator.evaluate(&batch, &mut answers);
//!         for (leaf, evaluation) in leaves.drain(..).zip(answers.drain(..)) {
//!             session.resume(leaf, evaluation);
//!         }
//!     }
//!     let decision = session.take_decision().expect("the session is decided");
//!     game.submit(generation, Reply::Place(decision)).expect("a fresh generation");
//! };
//! assert!(result.is_contested());
//! ```

pub mod gumbel;
pub mod mcts;
pub mod policy;
pub mod rng;
pub mod seam;
pub mod select;
pub mod session;

pub use gumbel::{GumbelConfig, GumbelSession, GumbelTrace};
pub use mcts::{MctsConfig, MctsSession};
pub use policy::PolicySession;
pub use rng::SplitMix64;
pub use seam::{EncodedBatch, Encoder, Evaluation, Evaluator};
pub use select::{Child, SearchOutcome, SelectFromPolicy, SelectFromSearch};
pub use session::{DecisionSession, LeafId, SessionStatus};
