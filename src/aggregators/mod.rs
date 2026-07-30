//! Byzantine-tolerant aggregation algorithms for federated learning.
//!
//! Provides multiple aggregation strategies with varying levels of
//! Byzantine tolerance:
//!
//! | Method | Assumption for robustness | Speed |
//! |--------|-------------------|-------|
//! | [`trimmed_mean`] | depends on trim fraction and attack model | Fast (parallel) |
//! | [`median`] | strictly <50% adversarial per coordinate | Fast (parallel) |
//! | [`krum`] | n >= 2f+3 | O(n^2) |
//! | Multi-Krum | n >= 2f+3, 1 <= m <= n-2f-2 | O(n^2) |
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
use crate::math::norms::l2_distance_f64;
use crate::reputation::validate::validate_threshold;
use crate::reputation::ReputationStore;
use crate::validation::{validate_client_ids, validate_updates};
use crate::verification::krum_condition::{krum_condition_met, krum_min_clients, max_multi_krum_m};

/// Vectors Multi-Krum selects when `m` is omitted, before capping to the safe
/// maximum. Chosen to preserve the historical default for cohorts large enough
/// to support it.
const DEFAULT_MULTI_KRUM_M: usize = 3;

/// Distance below which a client is treated as agreeing with the aggregate.
const REWARD_DISTANCE: f64 = 1.0;

/// Distance above which a client is treated as deviating from the aggregate.
const PENALTY_DISTANCE: f64 = 10.0;

/// Reputation gained by a client close to the aggregate.
const CLOSE_UPDATE_REWARD: f32 = 0.02;

/// Reputation lost by a client far from the aggregate.
const FAR_UPDATE_PENALTY: f32 = 0.08;

// Both adjustments feed validated store operations. Checked at compile time so
// the `?` on those calls is provably unreachable rather than merely unlikely.
const _: () = assert!(CLOSE_UPDATE_REWARD >= 0.0 && CLOSE_UPDATE_REWARD <= 1.0);
const _: () = assert!(FAR_UPDATE_PENALTY >= 0.0 && FAR_UPDATE_PENALTY <= 1.0);

/// Reject a persisted ban threshold that would not be accepted at construction.
fn deserialize_ban_threshold<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;

    let value = f32::deserialize(deserializer)?;
    validate_threshold(value)
        .map_err(|e| D::Error::custom(format!("{}", e)))
        .map(|_| value)
}

/// Aggregation method selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Coordinate-wise trimmed mean (default). Robustness depends on the trim
    /// fraction and attack model; see [`trimmed_mean`].
    TrimmedMean,
    /// Coordinate-wise median (~50% Byzantine tolerance)
    Median,
    /// Standard FedAvg (no Byzantine tolerance, baseline)
    FedAvg,
    /// Krum selection via BFP-16 block floating-point (deterministic, n >= 2f+3)
    ///
    /// The inner value is `f`, the maximum number of Byzantine nodes expected.
    /// Requires `n >= 2f + 3` clients; rounds below that are refused with
    /// [`QoraError::InsufficientQuorum`] rather than aggregated.
    Krum(usize),
    /// Multi-Krum: select top-m vectors by Krum score and average them.
    ///
    /// `(f, m)` — `f` is the max Byzantine count, `m` is the number of vectors
    /// to select and average. Requires `n >= 2f + 3` and `1 <= m <= n - 2f - 2`.
    ///
    /// `m` is deliberately an `Option` so that the aggregation boundary can
    /// tell an omitted `m` from an explicitly requested one:
    ///
    /// * `None` — cap to the safe maximum, `min(3, n - 2f - 2)`. There is no
    ///   caller intent to contradict, so a small cohort silently selects fewer
    ///   vectors instead of failing.
    /// * `Some(m)` — honored exactly, or refused with
    ///   [`QoraError::InvalidMultiKrumSelection`]. An explicit request is
    ///   never silently rewritten.
    MultiKrum(usize, Option<usize>),
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
    ///
    /// Validated on deserialization as well as on construction: restoring a
    /// persisted aggregator is a configuration path like any other, and a
    /// non-finite threshold here silently disables gating.
    #[serde(deserialize_with = "deserialize_ban_threshold")]
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
    /// * `trim_fraction` - Fraction trimmed from each end (0.0..=0.5). This is
    ///   a trimming parameter, not a tolerated attacker percentage.
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
    ///
    /// Gating fails closed: a round in which no client clears the threshold
    /// returns [`QoraError::AllUpdatesRejected`] rather than falling back to
    /// the full cohort. Note that unknown clients start at
    /// [`crate::reputation::store::DEFAULT_SCORE`] (0.5), so a threshold above
    /// that rejects every first-time participant -- which is now an error
    /// rather than a silently ungated round.
    ///
    /// # Errors
    ///
    /// [`QoraError::InvalidReputationThreshold`] if `ban_threshold` is not
    /// finite and within `[0.0, 1.0]`. This is not cosmetic: gating activates
    /// on `ban_threshold > 0.0`, which is false for NaN, so a non-finite
    /// threshold used to disable the gate entirely while appearing configured.
    pub fn with_ban_threshold(
        method: AggregationMethod,
        trim_fraction: f32,
        ban_threshold: f32,
    ) -> Result<Self, QoraError> {
        validate_threshold(ban_threshold)?;
        Ok(Self {
            method,
            trim_fraction,
            reputation: ReputationStore::new(),
            ban_threshold,
            adaptive_trim: false,
        })
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
    ///
    /// If gating rejects *every* submitted client the round fails with
    /// [`QoraError::AllUpdatesRejected`]. Rejected clients are never
    /// automatically restored: a client the configured policy excluded stays
    /// excluded even when no other client qualifies either.
    ///
    /// # Arguments
    ///
    /// * `updates` - Client model updates as 2D arrays
    /// * `client_ids` - Optional client identifiers for reputation tracking
    ///
    /// # Errors
    ///
    /// Checked in this order, so malformed input is never masked by a policy
    /// rejection:
    ///
    /// * [`QoraError::EmptyUpdates`] if `updates` is empty.
    /// * [`QoraError::DimensionMismatch`] if update shapes disagree.
    /// * [`QoraError::NonFiniteValue`] if any update contains NaN or infinity.
    /// * [`QoraError::ClientIdCountMismatch`] if `client_ids` is supplied and
    ///   its length differs from `updates`.
    /// * [`QoraError::AllUpdatesRejected`] if reputation gating leaves no
    ///   client.
    /// * [`QoraError::InsufficientQuorum`] for Krum and Multi-Krum when
    ///   `n < 2f + 3`.
    pub fn aggregate(
        &mut self,
        updates: &[Array2<f32>],
        client_ids: Option<&[String]>,
    ) -> Result<Array2<f32>, QoraError> {
        self.aggregate_weighted(updates, client_ids, None)
    }

    /// Aggregate with per-client weights.
    ///
    /// Identical to [`Self::aggregate`] except that `weights` (typically each
    /// client's sample count) participate in the mean.
    ///
    /// Weights apply to [`AggregationMethod::FedAvg`] only. The robust methods
    /// operate on client updates rather than per-sample votes, so reweighting
    /// them would change their Byzantine guarantee -- an attacker who claims a
    /// large sample count would gain proportional influence over a median or
    /// trimmed mean. Supplying weights for any other method is therefore an
    /// error rather than a silent no-op.
    ///
    /// Weights are filtered alongside updates when reputation gating removes a
    /// client, so the correspondence is preserved.
    ///
    /// # Errors
    ///
    /// Everything [`Self::aggregate`] can return, plus:
    ///
    /// * [`QoraError::WeightsNotSupported`] if weights accompany a method
    ///   other than FedAvg.
    /// * [`QoraError::NonFiniteWeight`] if any weight is NaN or infinite.
    /// * [`QoraError::DimensionMismatch`] if the weight count differs from the
    ///   update count.
    pub fn aggregate_weighted(
        &mut self,
        updates: &[Array2<f32>],
        client_ids: Option<&[String]>,
        weights: Option<&[f32]>,
    ) -> Result<Array2<f32>, QoraError> {
        if weights.is_some() && !matches!(self.method, AggregationMethod::FedAvg) {
            return Err(QoraError::WeightsNotSupported {
                method: format!("{:?}", self.method),
            });
        }

        // Validate before anything else. Two reasons this must come first:
        // the ban filter below indexes `updates` with positions derived from
        // `client_ids` (a length mismatch would panic), and non-finite values
        // are invisible to reputation tracking -- a NaN distance satisfies
        // neither the reward nor the penalty branch, so a NaN attacker would
        // otherwise be both unpenalized and unbannable.
        validate_updates(updates)?;
        validate_client_ids(updates, client_ids)?;
        if let Some(w) = weights {
            if w.len() != updates.len() {
                return Err(QoraError::DimensionMismatch);
            }
            crate::validation::validate_weights(w)?;
        }

        // Filter banned clients if reputation gating is enabled
        #[allow(clippy::type_complexity)]
        let (filtered_updates, filtered_ids, filtered_weights): (
            Vec<Array2<f32>>,
            Option<Vec<String>>,
            Option<Vec<f32>>,
        ) = if self.ban_threshold > 0.0 {
            if let Some(ids) = client_ids {
                let active: Vec<usize> = ids
                    .iter()
                    .enumerate()
                    .filter(|(_, id)| self.get_reputation(id) >= self.ban_threshold)
                    .map(|(i, _)| i)
                    .collect();

                // Fail closed. Reinstating the rejected clients here would
                // defeat the gate exactly when reputation distrusts the
                // whole cohort -- the one round where the policy matters
                // most. Returning before any aggregation also means no
                // score moves on this path.
                if active.is_empty() {
                    return Err(QoraError::AllUpdatesRejected {
                        total: updates.len(),
                        rejected: ids.len() - active.len(),
                        threshold: self.ban_threshold,
                    });
                }

                let u: Vec<Array2<f32>> = active.iter().map(|&i| updates[i].clone()).collect();
                let new_ids: Vec<String> = active.iter().map(|&i| ids[i].clone()).collect();
                // Weights are indexed by the same positions, so they must
                // be filtered with the same selection or they would be
                // applied to the wrong clients.
                let w = weights.map(|w| active.iter().map(|&i| w[i]).collect());
                (u, Some(new_ids), w)
            } else {
                (updates.to_vec(), None, weights.map(|w| w.to_vec()))
            }
        } else {
            (
                updates.to_vec(),
                client_ids.map(|ids| ids.to_vec()),
                weights.map(|w| w.to_vec()),
            )
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
            AggregationMethod::FedAvg => fedavg(agg_updates, filtered_weights.as_deref())?,
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
                        needed: krum_min_clients(f),
                        actual: agg_updates.len(),
                    })?;

                agg_updates[best_idx].clone()
            }
            AggregationMethod::MultiKrum(f, m) => {
                let n = agg_updates.len();

                // Quorum first: below it there is no safe m at all, and
                // reporting a maximum of 0 would be less actionable than
                // naming the real shortfall.
                if !krum_condition_met(n, f) {
                    return Err(QoraError::InsufficientQuorum {
                        needed: krum_min_clients(f),
                        actual: n,
                    });
                }

                let max_m = max_multi_krum_m(n, f);
                let m_eff = match m {
                    // Omitted: cap to the safe maximum. Preserves the
                    // historical m=3 wherever it is valid, and degrades to
                    // fewer selections for small cohorts rather than failing.
                    None => DEFAULT_MULTI_KRUM_M.min(max_m),
                    // Explicit: honored or refused, never rewritten.
                    Some(requested) if requested >= 1 && requested <= max_m => requested,
                    Some(requested) => {
                        return Err(QoraError::InvalidMultiKrumSelection {
                            clients: n,
                            byzantine: f,
                            selected: requested,
                            maximum: max_m,
                        })
                    }
                };

                let bfp_vecs: Vec<krum::Bfp16Vec> = agg_updates
                    .iter()
                    .map(|u| {
                        let flat: Vec<f32> = u.iter().copied().collect();
                        krum::Bfp16Vec::from_f32_slice(&flat)
                    })
                    .collect();

                let indices = aggregate_multi_krum_bfp16(&bfp_vecs, f, m_eff).ok_or(
                    QoraError::InsufficientQuorum {
                        needed: krum_min_clients(f),
                        actual: n,
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
            self.update_reputations(ids, agg_updates, &result)?;
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
    /// * `rate` - Decay rate in `[0.0, 1.0]`. Typical: 0.01-0.05 per round.
    ///
    /// # Errors
    ///
    /// [`QoraError::InvalidReputationDecay`] if `rate` is not finite and
    /// within `[0.0, 1.0]`. No score changes when the rate is rejected.
    pub fn decay_reputations(&mut self, rate: f32) -> Result<(), QoraError> {
        self.reputation.decay_toward_default(rate)
    }

    /// Get the ban threshold for this aggregator.
    pub fn ban_threshold(&self) -> f32 {
        self.ban_threshold
    }

    /// Update reputations based on how close each client's update is to the aggregate.
    ///
    /// The distance is accumulated in `f64` with operands widened before
    /// subtraction. In `f32` it had both failure directions: large finite
    /// differences overflowed to infinity and triggered a penalty based on the
    /// overflow rather than the actual deviation, while tiny differences
    /// underflowed to zero and earned a reward for a distance that was never
    /// measured.
    ///
    /// # Errors
    ///
    /// [`QoraError::NonFiniteReputationDistance`] if a distance is not finite.
    /// Updates are validated finite and the accumulation is `f64`, so this
    /// should be unreachable; it is reported rather than being treated as zero
    /// distance, which would read as maximum trust.
    fn update_reputations(
        &mut self,
        client_ids: &[String],
        updates: &[Array2<f32>],
        result: &Array2<f32>,
    ) -> Result<(), QoraError> {
        for (id, update) in client_ids.iter().zip(updates.iter()) {
            let distance = l2_distance_f64(update.iter(), result.iter());

            if !distance.is_finite() {
                return Err(QoraError::NonFiniteReputationDistance {
                    client_id: id.clone(),
                    value: distance,
                });
            }

            // Thresholds compared in f64 so the widened distance is not
            // narrowed back down before the decision.
            if distance < REWARD_DISTANCE {
                self.reputation.reward(id.clone(), CLOSE_UPDATE_REWARD)?;
            } else if distance > PENALTY_DISTANCE {
                self.reputation.penalize(id.clone(), FAR_UPDATE_PENALTY)?;
            }
        }
        Ok(())
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
    fn test_weighted_fedavg_honors_weights() {
        let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0);
        let updates = vec![array![[0.0]], array![[10.0]]];

        // Equal weighting would give 5.0.
        let result = agg
            .aggregate_weighted(&updates, None, Some(&[1.0, 9.0]))
            .unwrap();
        assert!(
            (result[[0, 0]] - 9.0).abs() < 1e-5,
            "got {}",
            result[[0, 0]]
        );
    }

    #[test]
    fn test_weights_are_refused_for_robust_methods() {
        // Reweighting a median by claimed sample count would let an attacker
        // buy influence with a number they control, so it is an error rather
        // than a silent no-op.
        for method in [
            AggregationMethod::Median,
            AggregationMethod::TrimmedMean,
            AggregationMethod::Krum(1),
            AggregationMethod::MultiKrum(1, None),
        ] {
            let mut agg = ByzantineAggregator::new(method.clone(), 0.2);
            let updates = vec![array![[1.0]], array![[2.0]]];
            assert!(
                matches!(
                    agg.aggregate_weighted(&updates, None, Some(&[1.0, 2.0])),
                    Err(QoraError::WeightsNotSupported { .. })
                ),
                "{:?} must refuse weights",
                method
            );
        }
    }

    #[test]
    fn test_weight_count_must_match_updates() {
        let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0);
        let updates = vec![array![[1.0]], array![[2.0]]];
        assert!(matches!(
            agg.aggregate_weighted(&updates, None, Some(&[1.0])),
            Err(QoraError::DimensionMismatch)
        ));
    }

    #[test]
    fn test_non_finite_weights_are_rejected() {
        let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0);
        let updates = vec![array![[1.0]], array![[2.0]]];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                matches!(
                    agg.aggregate_weighted(&updates, None, Some(&[1.0, bad])),
                    Err(QoraError::NonFiniteWeight { index: 1, .. })
                ),
                "weight {} must be rejected",
                bad
            );
        }
    }

    #[test]
    fn test_weights_follow_clients_through_reputation_gating() {
        // "banned" is filtered out; its weight must go with it rather than
        // being applied to whichever client takes its position.
        let json = r#"{"method":"FedAvg","trim_fraction":0.0,"reputation":{"banned":0.0,"good1":0.9,"good2":0.9},"ban_threshold":0.5,"adaptive_trim":false}"#;
        let mut agg: ByzantineAggregator = serde_json::from_str(json).unwrap();

        let updates = vec![array![[1000.0]], array![[0.0]], array![[10.0]]];
        let ids = vec![
            "banned".to_string(),
            "good1".to_string(),
            "good2".to_string(),
        ];

        let result = agg
            .aggregate_weighted(&updates, Some(&ids), Some(&[999.0, 1.0, 9.0]))
            .unwrap();

        // Only good1 (0.0, weight 1) and good2 (10.0, weight 9) survive: 9.0.
        assert!(
            (result[[0, 0]] - 9.0).abs() < 1e-4,
            "got {}",
            result[[0, 0]]
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
