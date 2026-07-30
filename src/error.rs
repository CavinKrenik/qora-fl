//! Error types for Qora-FL

use thiserror::Error;

/// All possible errors in Qora-FL
#[derive(Error, Debug)]
pub enum QoraError {
    /// No updates were provided for aggregation
    #[error("Empty updates provided")]
    EmptyUpdates,

    /// Updates have inconsistent dimensions
    #[error("Dimension mismatch in updates")]
    DimensionMismatch,

    /// Trim fraction is outside valid range
    #[error("Invalid trim fraction: {0} (must be 0.0-0.5)")]
    InvalidTrimFraction(f32),

    /// Not enough participants for quorum
    #[error("Insufficient quorum: need {needed}, got {actual}")]
    InsufficientQuorum {
        /// Minimum required participants
        needed: usize,
        /// Actual participants received
        actual: usize,
    },

    /// An update contains a non-finite value (NaN or infinity)
    ///
    /// `update_index` is positional within the `updates` slice. The free
    /// aggregation functions have no notion of client identity; the
    /// [`crate::ByzantineAggregator`] boundary can map the index back to a
    /// client ID if that context is needed.
    #[error("Non-finite value {value} in update {update_index} at [{row}, {col}]")]
    NonFiniteValue {
        /// Index of the offending update within the `updates` slice
        update_index: usize,
        /// Row of the offending value within that update
        row: usize,
        /// Column of the offending value within that update
        col: usize,
        /// The offending value (NaN, inf, or -inf)
        value: f32,
    },

    /// A client weight is non-finite (NaN or infinity)
    ///
    /// Updates can all be finite while a single non-finite weight corrupts
    /// the entire weighted mean.
    #[error("Non-finite weight {value} at index {index}")]
    NonFiniteWeight {
        /// Index of the offending weight within the `weights` slice
        index: usize,
        /// The offending value (NaN, inf, or -inf)
        value: f32,
    },

    /// The number of client IDs does not match the number of updates
    #[error("Client ID count mismatch: {updates} updates but {client_ids} client IDs")]
    ClientIdCountMismatch {
        /// Number of updates received
        updates: usize,
        /// Number of client IDs received
        client_ids: usize,
    },

    /// Error in reputation tracking
    #[error("Reputation error: {0}")]
    ReputationError(String),

    /// Verification check failed
    #[error("Verification failed: {0}")]
    VerificationError(String),

    /// A norm bound is not a usable threshold
    ///
    /// A bound must be finite and strictly positive. Zero and negative bounds
    /// reject every possible update; an infinite bound accepts every update
    /// regardless of its norm; a NaN bound rejects every update because every
    /// comparison against NaN is false. All four are caller misconfiguration,
    /// not verification failures, and are reported separately so that a caller
    /// can distinguish an unusable threshold from a genuinely oversized update.
    #[error("Invalid norm bound: {value} (must be finite and > 0)")]
    InvalidNormBound {
        /// The offending bound
        value: f32,
    },

    /// Array shape mismatch
    #[error("Array shape error: {0}")]
    ShapeError(String),
}

impl From<ndarray::ShapeError> for QoraError {
    fn from(e: ndarray::ShapeError) -> Self {
        QoraError::ShapeError(e.to_string())
    }
}
