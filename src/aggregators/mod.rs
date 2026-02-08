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
//! | Multi-Krum | n >= 2f+3, m <= n-2f-2 | O(n^2) |
//! | [`fedavg`] | None (baseline) | Fastest |

pub mod adaptive;
pub mod fedavg;
pub mod krum;
pub mod median;
pub mod trimmed_mean;

pub use fedavg::fedavg;
pub use krum::aggregate_krum;
pub use krum::aggregate_krum_bfp16;
pub use krum::aggregate_multi_krum_bfp16;
pub use median::median;
pub use trimmed_mean::trimmed_mean;

use ndarray::Array2;

use serde::{Deserialize, Serialize};

use crate::error::QoraError;
use crate::reputation::ReputationStore;

/// Aggregation method selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Coordinate-wise trimmed mean (default, ~30% Byzantine tolerance)
    TrimmedMean,
    /// Coordinate-wise median (~50% Byzantine tolerance)
    Median,
    /// Standard FedAvg (no Byzantine tolerance, baseline)
    FedAvg,
    /// Krum selection via BFP-16 block floating-point (deterministic, n >= 2f+3)
    ///
    /// The inner value is `f`, the maximum number of Byzantine nodes expected.
    /// Requires `n >= 2f + 3` clients for full guarantees (best-effort below).
    Krum(usize),
    /// Multi-Krum: select top-m vectors by Krum score and average them.
    ///
    /// `(f, m)` — `f` is the max Byzantine count, `m` is the number of vectors
    /// to select and average. For full tolerance, `m <= n - 2f - 2`.
    MultiKrum(usize, usize),
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
#[derive(Serialize, Deserialize)]
pub struct ByzantineAggregator {
    method: AggregationMethod,
    trim_fraction: f32,
    reputation: ReputationStore<String>,
    /// Clients below this reputation score are excluded from aggregation.
    /// Default: 0.0 (no gating). Set to e.g. 0.2 to enable ban gating.
    ban_threshold: f32,
    /// When true and method is TrimmedMean, compute trim_fraction dynamically
    /// from client reputation distribution each round.
    #[serde(default)]
    adaptive_trim: bool,
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
            reputation: ReputationStore::new(),
            ban_threshold: 0.0,
            adaptive_trim: false,
        }
    }

    /// Create a new aggregator with reputation-based gating.
    ///
    /// Clients whose reputation falls below `ban_threshold` are excluded
    /// from aggregation when `client_ids` are provided.
    pub fn with_ban_threshold(
        method: AggregationMethod,
        trim_fraction: f32,
        ban_threshold: f32,
    ) -> Self {
        Self {
            method,
            trim_fraction,
            reputation: ReputationStore::new(),
            ban_threshold,
            adaptive_trim: false,
        }
    }

    /// Enable or disable adaptive trimming.
    ///
    /// When enabled and method is `TrimmedMean`, the trim fraction is computed
    /// dynamically each round from the client reputation distribution.
    pub fn set_adaptive_trim(&mut self, enabled: bool) {
        self.adaptive_trim = enabled;
    }

    /// Aggregate client model updates.
    ///
    /// When `client_ids` are provided and `ban_threshold > 0`, clients whose
    /// reputation is below the threshold are excluded before aggregation.
    /// If all clients would be excluded, the filter is bypassed (fail-open).
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
        // Filter banned clients if reputation gating is enabled
        let (filtered_updates, filtered_ids): (Vec<Array2<f32>>, Option<Vec<String>>) =
            if self.ban_threshold > 0.0 {
                if let Some(ids) = client_ids {
                    let active: Vec<usize> = ids
                        .iter()
                        .enumerate()
                        .filter(|(_, id)| self.get_reputation(id) >= self.ban_threshold)
                        .map(|(i, _)| i)
                        .collect();

                    if active.is_empty() {
                        // Fail-open: use all clients rather than empty aggregation
                        (updates.to_vec(), Some(ids.to_vec()))
                    } else {
                        let u: Vec<Array2<f32>> =
                            active.iter().map(|&i| updates[i].clone()).collect();
                        let new_ids: Vec<String> = active.iter().map(|&i| ids[i].clone()).collect();
                        (u, Some(new_ids))
                    }
                } else {
                    (updates.to_vec(), None)
                }
            } else {
                (updates.to_vec(), client_ids.map(|ids| ids.to_vec()))
            };

        let agg_updates = &filtered_updates;

        let result = match self.method {
            AggregationMethod::TrimmedMean => {
                let frac = if self.adaptive_trim {
                    adaptive::compute_adaptive_trim(
                        self.reputation.scores(),
                        0.4,
                        0.05,
                        self.trim_fraction,
                    )
                } else {
                    self.trim_fraction
                };
                trimmed_mean(agg_updates, frac)?
            }
            AggregationMethod::Median => median(agg_updates)?,
            AggregationMethod::FedAvg => fedavg(agg_updates, None)?,
            AggregationMethod::Krum(f) => {
                let bfp_vecs: Vec<krum::Bfp16Vec> = agg_updates
                    .iter()
                    .map(|u| {
                        let flat: Vec<f32> = u.iter().copied().collect();
                        krum::Bfp16Vec::from_f32_slice(&flat)
                    })
                    .collect();

                let best_idx =
                    aggregate_krum_bfp16(&bfp_vecs, f).ok_or(QoraError::InsufficientQuorum {
                        needed: 3,
                        actual: agg_updates.len(),
                    })?;

                agg_updates[best_idx].clone()
            }
            AggregationMethod::MultiKrum(f, m) => {
                let bfp_vecs: Vec<krum::Bfp16Vec> = agg_updates
                    .iter()
                    .map(|u| {
                        let flat: Vec<f32> = u.iter().copied().collect();
                        krum::Bfp16Vec::from_f32_slice(&flat)
                    })
                    .collect();

                let indices = aggregate_multi_krum_bfp16(&bfp_vecs, f, m).ok_or(
                    QoraError::InsufficientQuorum {
                        needed: 3,
                        actual: agg_updates.len(),
                    },
                )?;

                // Average the selected original f32 vectors
                let m_eff = indices.len() as f32;
                let mut avg = Array2::<f32>::zeros(agg_updates[0].raw_dim());
                for &idx in &indices {
                    avg += &agg_updates[idx];
                }
                avg /= m_eff;
                avg
            }
        };

        // Track reputation based on distance to aggregate
        if let Some(ref ids) = filtered_ids {
            self.update_reputations(ids, agg_updates, &result);
        }

        Ok(result)
    }

    /// Get the reputation score for a client (default 0.5 for unknown clients).
    pub fn get_reputation(&self, client_id: &str) -> f32 {
        self.reputation.get_score(client_id)
    }

    /// Reset all reputation scores.
    pub fn reset_reputation(&mut self) {
        self.reputation.clear();
    }

    /// Decay all reputation scores toward the default (0.5).
    ///
    /// Call once per round to allow penalized clients to recover over time
    /// and prevent stale high-reputation scores from persisting indefinitely.
    ///
    /// # Arguments
    ///
    /// * `rate` - Decay rate in (0.0, 1.0). Typical: 0.01-0.05 per round.
    pub fn decay_reputations(&mut self, rate: f32) {
        self.reputation.decay_toward_default(rate);
    }

    /// Get the ban threshold for this aggregator.
    pub fn ban_threshold(&self) -> f32 {
        self.ban_threshold
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

            if distance < 1.0 {
                self.reputation.reward(id.clone(), 0.02);
            } else if distance > 10.0 {
                self.reputation.penalize(id.clone(), 0.08);
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

    #[test]
    fn test_aggregator_krum_selects_honest() {
        // f=1 Byzantine, need n >= 2*1+3 = 5 clients
        let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);

        let updates = vec![
            array![[1.0, 1.0]],
            array![[1.1, 0.9]],
            array![[0.9, 1.1]],
            array![[1.05, 0.95]],
            array![[100.0, 100.0]], // Byzantine
        ];

        let result = agg.aggregate(&updates, None).unwrap();
        assert!(
            result[[0, 0]] < 2.0,
            "Krum should select honest vector, got {}",
            result[[0, 0]]
        );
    }

    #[test]
    fn test_aggregator_krum_deterministic() {
        let updates = vec![
            array![[1.0, 2.0]],
            array![[1.1, 2.1]],
            array![[0.9, 1.9]],
            array![[1.05, 2.05]],
            array![[50.0, 50.0]],
        ];

        let mut agg1 = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);
        let mut agg2 = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);

        let r1 = agg1.aggregate(&updates, None).unwrap();
        let r2 = agg2.aggregate(&updates, None).unwrap();
        assert_eq!(
            r1, r2,
            "Krum through ByzantineAggregator must be deterministic"
        );
    }

    #[test]
    fn test_aggregator_krum_too_few_clients() {
        let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);

        // Only 2 clients -- Krum needs at least 3
        let updates = vec![array![[1.0]], array![[2.0]]];
        let result = agg.aggregate(&updates, None);
        assert!(result.is_err(), "Krum with <3 clients should error");
    }
}
