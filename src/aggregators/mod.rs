//! Byzantine-tolerant aggregation algorithms for federated learning.
//!
//! Provides multiple aggregation strategies with varying levels of
//! Byzantine tolerance:
//!
//! | Method | Byzantine Tolerance | Speed |
//! |--------|-------------------|-------|
//! | [`trimmed_mean`] | ~30% | Fast (parallel) |
//! | [`median`] | ~50% | Fast (parallel) |
//! | [`krum`] | n >= 2f+3 | O(n^2) |
//! | [`fedavg`] | None (baseline) | Fastest |

pub mod fedavg;
pub mod krum;
pub mod median;
pub mod trimmed_mean;

pub use fedavg::fedavg;
pub use krum::aggregate_krum;
pub use median::median;
pub use trimmed_mean::trimmed_mean;

use ndarray::Array2;
use std::collections::HashMap;

use crate::error::QoraError;

/// Aggregation method selection.
#[derive(Clone, Debug, PartialEq)]
pub enum AggregationMethod {
    /// Coordinate-wise trimmed mean (default, ~30% Byzantine tolerance)
    TrimmedMean,
    /// Coordinate-wise median (~50% Byzantine tolerance)
    Median,
    /// Standard FedAvg (no Byzantine tolerance, baseline)
    FedAvg,
}

/// High-level Byzantine-tolerant aggregator for federated learning.
///
/// Wraps the individual aggregation functions with optional client
/// reputation tracking.
///
/// # Example
///
/// ```rust
/// use qora_fl::ByzantineAggregator;
/// use qora_fl::aggregators::AggregationMethod;
/// use ndarray::array;
///
/// let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);
///
/// let updates = vec![
///     array![[1.0, 2.0]],
///     array![[1.1, 2.1]],
///     array![[1.0, 1.9]],
///     array![[100.0, 200.0]], // Byzantine
/// ];
///
/// let result = agg.aggregate(&updates, None).unwrap();
/// assert!((result[[0, 0]] - 1.0).abs() < 0.5);
/// ```
pub struct ByzantineAggregator {
    method: AggregationMethod,
    trim_fraction: f32,
    reputation: HashMap<String, f32>,
}

impl ByzantineAggregator {
    /// Create a new aggregator.
    ///
    /// # Arguments
    ///
    /// * `method` - Which aggregation algorithm to use
    /// * `trim_fraction` - Fraction to trim from each end (0.0..0.5, typically 0.2 for 30% tolerance).
    ///   Only used by [`AggregationMethod::TrimmedMean`].
    pub fn new(method: AggregationMethod, trim_fraction: f32) -> Self {
        Self {
            method,
            trim_fraction,
            reputation: HashMap::new(),
        }
    }

    /// Aggregate client model updates.
    ///
    /// # Arguments
    ///
    /// * `updates` - Client model updates as 2D arrays
    /// * `client_ids` - Optional client identifiers for reputation tracking
    pub fn aggregate(
        &mut self,
        updates: &[Array2<f32>],
        client_ids: Option<&[String]>,
    ) -> Result<Array2<f32>, QoraError> {
        let result = match self.method {
            AggregationMethod::TrimmedMean => trimmed_mean(updates, self.trim_fraction)?,
            AggregationMethod::Median => median(updates)?,
            AggregationMethod::FedAvg => fedavg(updates, None)?,
        };

        // Track reputation if client IDs provided
        if let Some(ids) = client_ids {
            self.update_reputations(ids, updates, &result);
        }

        Ok(result)
    }

    /// Get the reputation score for a client (default 0.5 for unknown clients).
    pub fn get_reputation(&self, client_id: &str) -> f32 {
        self.reputation.get(client_id).copied().unwrap_or(0.5)
    }

    /// Reset all reputation scores.
    pub fn reset_reputation(&mut self) {
        self.reputation.clear();
    }

    /// Update reputations based on how close each client's update is to the aggregate.
    fn update_reputations(
        &mut self,
        client_ids: &[String],
        updates: &[Array2<f32>],
        result: &Array2<f32>,
    ) {
        for (id, update) in client_ids.iter().zip(updates.iter()) {
            let diff = update - result;
            let distance: f32 = diff.iter().map(|x| x * x).sum::<f32>().sqrt();

            let score = self.reputation.entry(id.clone()).or_insert(0.5);
            if distance < 1.0 {
                // Close to aggregate -> reward
                *score = (*score + 0.02).min(1.0);
            } else if distance > 10.0 {
                // Far from aggregate -> penalize
                *score = (*score - 0.08).max(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_aggregator_trimmed_mean() {
        let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);

        let updates = vec![
            array![[1.0]],
            array![[1.1]],
            array![[0.9]],
            array![[1.0]],
            array![[100.0]], // Byzantine
        ];

        let result = agg.aggregate(&updates, None).unwrap();
        assert!(result[[0, 0]] < 2.0, "Should reject outlier");
    }

    #[test]
    fn test_aggregator_median() {
        let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0);

        let updates = vec![array![[1.0]], array![[2.0]], array![[100.0]]];

        let result = agg.aggregate(&updates, None).unwrap();
        assert_eq!(result[[0, 0]], 2.0);
    }

    #[test]
    fn test_aggregator_fedavg() {
        let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0);

        let updates = vec![array![[1.0]], array![[3.0]]];

        let result = agg.aggregate(&updates, None).unwrap();
        assert!((result[[0, 0]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_reputation_tracking() {
        let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);

        let updates = vec![
            array![[1.0]],
            array![[1.0]],
            array![[1.0]],
            array![[1.0]],
            array![[100.0]], // Byzantine
        ];

        let ids: Vec<String> = (0..5).map(|i| format!("client_{}", i)).collect();

        let _ = agg.aggregate(&updates, Some(&ids)).unwrap();

        // Honest clients should have higher reputation than Byzantine
        assert!(agg.get_reputation("client_0") > agg.get_reputation("client_4"));
    }

    #[test]
    fn test_default_reputation() {
        let agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);
        assert_eq!(agg.get_reputation("unknown"), 0.5);
    }

    #[test]
    fn test_reset_reputation() {
        let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);

        let updates = vec![array![[1.0]], array![[1.0]], array![[1.0]]];
        let ids = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let _ = agg.aggregate(&updates, Some(&ids));

        agg.reset_reputation();
        assert_eq!(agg.get_reputation("a"), 0.5);
    }
}
