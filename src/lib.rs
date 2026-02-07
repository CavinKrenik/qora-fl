//! # Qora-FL: Quorum-Oriented Robust Aggregation for Federated Learning
//!
//! Qora (pronounced "KOR-ah") provides Byzantine-tolerant aggregation
//! for federated learning through quorum consensus.
//!
//! ## Aggregation Methods
//!
//! - [`trimmed_mean()`] - Coordinate-wise trimmed mean (~30% Byzantine tolerance)
//! - [`median()`] - Coordinate-wise median (~50% Byzantine tolerance)
//! - [`aggregate_krum()`] - Krum selection with fixed-point math (n >= 2f+3)
//! - [`fedavg()`] - Standard FedAvg baseline (no Byzantine tolerance)
//!
//! ## High-Level API
//!
//! Use [`ByzantineAggregator`] for a convenient interface with built-in
//! reputation tracking.

#![deny(missing_docs)]

pub mod aggregators;
pub mod error;
pub mod math;
pub mod reputation;
pub mod verification;

// Re-exports
pub use aggregators::aggregate_krum;
pub use aggregators::fedavg;
pub use aggregators::median;
pub use aggregators::trimmed_mean;
pub use aggregators::{AggregationMethod, ByzantineAggregator};
pub use error::QoraError;
pub use reputation::ReputationTracker;

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
