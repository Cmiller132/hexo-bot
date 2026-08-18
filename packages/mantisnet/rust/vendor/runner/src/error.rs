//! Submission refusal types.

/// A submission the game refused to act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SubmitError {
    /// The game has already ended.
    Finished,
    /// The submission answers a decision the game has moved past.
    StaleGeneration {
        /// The token the game is waiting on.
        expected: u64,
        /// The token that was submitted.
        got: u64,
    },
    /// The seat's mirror disagrees with the canonical position.
    Desync {
        /// The canonical hash.
        expected: u64,
        /// The hash the seat believed.
        got: u64,
    },
}

impl core::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Finished => f.write_str("the game has already ended"),
            Self::StaleGeneration { expected, got } => write!(
                f,
                "submission is for generation {got}, but the game is at {expected}"
            ),
            Self::Desync { expected, got } => write!(
                f,
                "the seat's position hash is {got:#018x}, but the canonical one is {expected:#018x}"
            ),
        }
    }
}

impl core::error::Error for SubmitError {}
