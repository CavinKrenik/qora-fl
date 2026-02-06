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
    
    /// Error in reputation tracking
    #[error("Reputation error: {0}")]
    ReputationError(String),
    
    /// Verification check failed
    #[error("Verification failed: {0}")]
    VerificationError(String),
    
    /// Array shape mismatch
    #[error("Array shape error: {0}")]
    ShapeError(String),
}

impl From<ndarray::ShapeError> for QoraError {
    fn from(e: ndarray::ShapeError) -> Self {
        QoraError::ShapeError(e.to_string())
    }
}
