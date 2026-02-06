//! # Qora-FL: Quorum-Oriented Robust Aggregation for Federated Learning
//!
//! Qora (pronounced "KOR-ah") provides Byzantine-tolerant aggregation
//! for federated learning through quorum consensus.

#![deny(missing_docs)]

pub mod aggregators;
pub mod reputation;
pub mod verification;
pub mod math;
pub mod error;

// Re-exports for convenience
pub use aggregators::aggregate_krum;
pub use reputation::ReputationTracker;
pub use error::QoraError;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
