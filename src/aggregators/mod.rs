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
//!
//! # Participation filtering
//!
//! [`ByzantineAggregator`] can exclude clients before aggregating, through two
//! independent and individually optional policies:
//!
//! * **Reputation gating** ([`ByzantineAggregator::with_ban_threshold`]) removes
//!   clients below a score threshold.
//! * **Norm-bound filtering**
//!   ([`ByzantineAggregator::with_norm_bound_filter`]) removes updates whose L2
//!   norm exceeds a caller-chosen bound.
//!
//! Both are off by default. Gating runs first and a client it removes is not
//! measured again, so every submitted update carries at most one rejection
//! reason. If nothing survives, the round fails closed with
//! [`QoraError::AllUpdatesRejected`] rather than reinstating anyone. Method
//! preconditions -- Krum's quorum, Multi-Krum's selection bound -- are enforced
//! against the *accepted* cohort, so filtering can legitimately turn a workable
//! round into a refused one.
//!
//! [`ByzantineAggregator::aggregate_with_audit`] returns the same result plus a
//! caller-owned record of what was decided about each update. See
//! [`ByzantineAggregator::aggregate`] for the full ordering.

pub mod adaptive;
pub mod audited;
pub mod fedavg;
pub mod krum;
pub mod median;
pub mod trimmed_mean;

pub use audited::{AuditedAggregation, AuditedAggregationError};
pub use fedavg::fedavg;
pub use krum::aggregate_krum;
pub use krum::aggregate_krum_bfp16;
pub use krum::aggregate_multi_krum_bfp16;
pub use median::median;
pub use trimmed_mean::trimmed_mean;

use ndarray::Array2;

use serde::{Deserialize, Serialize};

use crate::audit::{
    AggregationAuditDecision, AggregationAuditEntry, AggregationAuditOutcome, AggregationDecision,
    AggregationRejectionReason, AuditedAggregationMethod,
};
use crate::error::QoraError;
use crate::math::norms::l2_distance_f64;
use crate::reputation::validate::validate_threshold;
use crate::reputation::ReputationStore;
use crate::validation::{validate_client_ids, validate_updates};
use crate::verification::krum_condition::{krum_condition_met, krum_min_clients, max_multi_krum_m};
use crate::verification::norm_bound::{evaluate_norm_bound, validate_norm_bound};

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

/// Reject a persisted norm bound that would not be accepted at construction.
///
/// A missing field never reaches this function -- `#[serde(default)]` supplies
/// `None`, which is the disabled state and needs no validation. An explicit
/// `null` does reach it and is equally valid. Anything else must clear the same
/// bar as [`ByzantineAggregator::with_norm_bound_filter`]: restoring a persisted
/// aggregator is a configuration path like any other, and an unusable bound
/// stored on disk would otherwise reject every update at the next round.
fn deserialize_norm_bound<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;

    let value = Option::<f32>::deserialize(deserializer)?;
    if let Some(bound) = value {
        validate_norm_bound(bound).map_err(|e| D::Error::custom(format!("{}", e)))?;
    }
    Ok(value)
}

/// Encode updates for the Krum family's integer scoring.
fn encode_bfp16(updates: &[Array2<f32>]) -> Vec<krum::Bfp16Vec> {
    updates
        .iter()
        .map(|update| {
            let flat: Vec<f32> = update.iter().copied().collect();
            krum::Bfp16Vec::from_f32_slice(&flat)
        })
        .collect()
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
    /// Largest L2 norm an update may have and still participate, or `None` to
    /// disable norm filtering entirely. Default: `None`.
    ///
    /// `#[serde(default)]` so that a 0.3.1 configuration, written before the
    /// field existed, restores with filtering off rather than failing to
    /// deserialize. A bound that *is* present is validated on the way in.
    #[serde(default, deserialize_with = "deserialize_norm_bound")]
    norm_bound: Option<f32>,
}

/// One submitted update and everything that must travel with it.
///
/// Filtering removes a record, which removes its update, identity, weight, and
/// original position together. Maintaining those as parallel vectors and
/// indexing them separately is what previously let a rejected client's sample
/// weight land on whichever client took its place; the correspondence is
/// structural here rather than maintained by convention.
///
/// `original_index` is the position in the caller's slice and is never
/// renumbered by filtering, so every audit decision points back at what the
/// caller actually sent.
struct Candidate<'a> {
    original_index: usize,
    update: &'a Array2<f32>,
    client_id: Option<&'a str>,
    weight: Option<f32>,
}

/// A round that produced an aggregate, with the material for its audit record.
struct RunOutcome {
    aggregate: Array2<f32>,
    method: AuditedAggregationMethod,
    decisions: Vec<AggregationAuditDecision>,
}

/// A round that produced no aggregate.
struct RunFailure {
    error: QoraError,
    /// Present only when filtering rejected every submitted update: the one
    /// failure that schema version 1 can describe, and the one where the
    /// per-update reasons are all the caller has to go on.
    rejected: Option<RejectedRound>,
}

/// The material for an all-rejected audit record.
struct RejectedRound {
    method: AuditedAggregationMethod,
    decisions: Vec<AggregationAuditDecision>,
}

impl From<QoraError> for RunFailure {
    fn from(error: QoraError) -> Self {
        Self {
            error,
            rejected: None,
        }
    }
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
            norm_bound: None,
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
            norm_bound: None,
        })
    }

    /// Enable or disable adaptive trimming.
    ///
    /// When enabled and method is `TrimmedMean`, the trim fraction is computed
    /// dynamically each round from the client reputation distribution.
    pub fn set_adaptive_trim(&mut self, enabled: bool) {
        self.adaptive_trim = enabled;
    }

    /// Exclude updates whose L2 norm exceeds `bound`.
    ///
    /// **Opt-in.** Filtering is off by default and stays off until this is
    /// called; an aggregator that never calls it computes no norms and behaves
    /// exactly as it did before the filter existed.
    ///
    /// The name says *filter* rather than *check* because that is what it does:
    /// an over-bound update is removed from the round, not merely reported. The
    /// norm is computed in `f64` and equality is acceptance -- a norm exactly
    /// equal to `bound` participates.
    ///
    /// # Choosing a bound
    ///
    /// The caller picks it, and there is no default worth having. The
    /// appropriate L2 bound depends on model scale, learning rate, local epoch
    /// count, and whether updates are deltas or full weights -- none of which
    /// this crate observes.
    ///
    /// A large norm is a **policy violation, not evidence of malice.** A client
    /// excluded by this filter has broken a limit the operator set; it has not
    /// been shown to be Byzantine, and its reputation is deliberately left
    /// untouched. See [`Self::aggregate`] for how filtering interacts with
    /// reputation gating and method quorum.
    ///
    /// # Example
    ///
    /// ```rust
    /// use qora_fl::ByzantineAggregator;
    /// use qora_fl::aggregators::AggregationMethod;
    /// use ndarray::array;
    ///
    /// let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0)
    ///     .with_norm_bound_filter(10.0)?;
    ///
    /// let updates = vec![
    ///     array![[1.0, 1.0]],
    ///     array![[1.0, 2.0]],
    ///     array![[900.0, 900.0]], // norm ~1273, excluded
    /// ];
    ///
    /// let result = agg.aggregate(&updates, None)?;
    /// assert_eq!(result[[0, 0]], 1.0);
    /// # Ok::<(), qora_fl::QoraError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`QoraError::InvalidNormBound`] if `bound` is not finite and strictly
    /// positive. Zero and negative bounds reject every update, an infinite bound
    /// accepts every update regardless of its norm, and NaN rejects every update
    /// because every comparison against it is false. None is clamped to
    /// something usable: a caller who passed one of them has misconfigured the
    /// filter and silently repairing it would hide that.
    pub fn with_norm_bound_filter(mut self, bound: f32) -> Result<Self, QoraError> {
        validate_norm_bound(bound)?;
        self.norm_bound = Some(bound);
        Ok(self)
    }

    /// Disable norm-bound filtering, restoring the unfiltered behavior.
    pub fn without_norm_bound_filter(mut self) -> Self {
        self.norm_bound = None;
        self
    }

    /// The configured norm bound, or `None` when filtering is disabled.
    pub fn norm_bound(&self) -> Option<f32> {
        self.norm_bound
    }

    /// Aggregate client model updates.
    ///
    /// # Filtering order
    ///
    /// A round proceeds in exactly this order, and the ordering is part of the
    /// contract:
    ///
    /// 1. The **complete submitted batch** is validated. Malformed input is an
    ///    error, never a filtering decision, so no client can hide a poisoned
    ///    update behind a policy rejection.
    /// 2. Every update starts accepted.
    /// 3. **Reputation gating** removes clients below `ban_threshold`, when
    ///    `client_ids` are supplied and the threshold is above zero.
    /// 4. **Norm-bound filtering** measures only the candidates gating left
    ///    active, when a bound is configured. A client already rejected is not
    ///    judged twice: the first policy to reject an update owns its reason.
    /// 5. If nothing remains the round **fails closed** with
    ///    [`QoraError::AllUpdatesRejected`]. Rejected clients are never
    ///    restored, no aggregation runs, and no reputation moves.
    /// 6. Effective method parameters are resolved against the **accepted
    ///    cohort**, and the method's preconditions are enforced against it.
    ///    Filtering can therefore shrink a valid cohort into one Krum or
    ///    Multi-Krum refuses; that is reported rather than repaired by
    ///    reinstating clients or weakening `f`.
    /// 7. Aggregation runs, then reputation is updated from the distance
    ///    between each **accepted** client's update and the aggregate.
    ///
    /// A norm rejection is a participation decision only. It does not reward,
    /// penalize, or ban the client, and the excluded update's weight leaves the
    /// round with it. With no bound configured (the default) no norm is
    /// computed and behavior is exactly as it was before the filter existed.
    ///
    /// This method reports only success or failure. Use
    /// [`Self::aggregate_with_audit`] to also receive the per-update decisions.
    ///
    /// # Arguments
    ///
    /// * `updates` - Client model updates as 2D arrays
    /// * `client_ids` - Optional client identifiers for reputation tracking
    ///
    /// # Errors
    ///
    /// Checked in the order above:
    ///
    /// * [`QoraError::EmptyUpdates`] if `updates` is empty.
    /// * [`QoraError::DimensionMismatch`] if update shapes disagree.
    /// * [`QoraError::NonFiniteValue`] if any update contains NaN or infinity.
    /// * [`QoraError::ClientIdCountMismatch`] if `client_ids` is supplied and
    ///   its length differs from `updates`.
    /// * [`QoraError::InvalidNormBound`] if a configured bound is not finite
    ///   and strictly positive.
    /// * [`QoraError::AllUpdatesRejected`] if filtering leaves no client. The
    ///   variant is deliberately generic: reputation gating, norm filtering, or
    ///   any mixture of the two can produce it.
    /// * [`QoraError::InsufficientQuorum`] for Krum and Multi-Krum when the
    ///   *accepted* cohort violates `n >= 2f + 3`.
    /// * [`QoraError::InvalidMultiKrumSelection`] for an explicit Multi-Krum
    ///   `m` outside the range the accepted cohort supports.
    pub fn aggregate(
        &mut self,
        updates: &[Array2<f32>],
        client_ids: Option<&[String]>,
    ) -> Result<Array2<f32>, QoraError> {
        self.aggregate_weighted(updates, client_ids, None)
    }

    /// Aggregate, and return the audit record alongside the aggregate.
    ///
    /// Same engine, same filtering, same errors as [`Self::aggregate`] -- the
    /// only difference is that the per-update decisions are kept instead of
    /// discarded. The record is handed over once and not retained: this
    /// aggregator stores no audit state and its serialized form is unchanged.
    ///
    /// The record survives the all-rejected failure, which is the round it
    /// exists for: [`AuditedAggregationError::audit`] carries every rejection
    /// reason even though no aggregate was produced. See that type for exactly
    /// which failures carry a record and which cannot.
    ///
    /// # Errors
    ///
    /// Everything [`Self::aggregate`] can return, wrapped in
    /// [`AuditedAggregationError`]. Use
    /// [`AuditedAggregationError::source_error`] to recover the underlying
    /// [`QoraError`].
    pub fn aggregate_with_audit(
        &mut self,
        updates: &[Array2<f32>],
        client_ids: Option<&[String]>,
    ) -> Result<AuditedAggregation, AuditedAggregationError> {
        self.aggregate_weighted_with_audit(updates, client_ids, None)
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
    /// Weights are filtered alongside updates when a client is removed -- by
    /// reputation gating or by the norm bound -- so the correspondence is
    /// preserved and a rejected client's weight never reaches the denominator.
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
        self.run(updates, client_ids, weights)
            .map(|outcome| outcome.aggregate)
            .map_err(|failure| failure.error)
    }

    /// Aggregate with per-client weights, returning the audit record too.
    ///
    /// The weighted counterpart of [`Self::aggregate_with_audit`]; see both for
    /// the semantics, which are otherwise identical.
    ///
    /// # Errors
    ///
    /// Everything [`Self::aggregate_weighted`] can return, wrapped in
    /// [`AuditedAggregationError`].
    pub fn aggregate_weighted_with_audit(
        &mut self,
        updates: &[Array2<f32>],
        client_ids: Option<&[String]>,
        weights: Option<&[f32]>,
    ) -> Result<AuditedAggregation, AuditedAggregationError> {
        match self.run(updates, client_ids, weights) {
            Ok(outcome) => {
                // Unreachable in practice: every descriptor the engine builds
                // describes a method that just ran successfully, and a method
                // with parameters the schema refuses cannot have run.
                let entry = AggregationAuditEntry::new(
                    outcome.method,
                    outcome.decisions,
                    AggregationAuditOutcome::Aggregated,
                )
                .map_err(AuditedAggregationError::bare)?;
                Ok(AuditedAggregation::new(outcome.aggregate, entry))
            }
            Err(failure) => Err(match failure.rejected {
                None => AuditedAggregationError::bare(failure.error),
                Some(rejected) => {
                    // A record that cannot be built must not replace the
                    // failure that produced it: the caller still needs to hear
                    // that the round was refused, so the entry is dropped and
                    // the error kept.
                    let entry = AggregationAuditEntry::new(
                        rejected.method,
                        rejected.decisions,
                        AggregationAuditOutcome::AllUpdatesRejected,
                    )
                    .ok();
                    AuditedAggregationError::with_audit(failure.error, entry)
                }
            }),
        }
    }

    /// The one aggregation engine.
    ///
    /// Every public aggregation method delegates here, so validation, filtering,
    /// parameter resolution, and reputation updates each happen exactly once and
    /// the audited and ordinary APIs cannot drift apart. The ordinary APIs
    /// discard [`RunOutcome::method`] and [`RunOutcome::decisions`]; that is the
    /// only difference between them.
    ///
    /// See [`Self::aggregate`] for the ordering this implements and why.
    fn run(
        &mut self,
        updates: &[Array2<f32>],
        client_ids: Option<&[String]>,
        weights: Option<&[f32]>,
    ) -> Result<RunOutcome, RunFailure> {
        // ---- 1. The complete submitted batch, before anyone is excluded ----
        //
        // Malformed input outranks every policy decision. A banned or oversized
        // client must not be able to conceal a poisoned batch behind its own
        // rejection, and non-finite values are invisible to reputation tracking
        // -- a NaN distance satisfies neither the reward nor the penalty branch,
        // leaving a NaN attacker unpenalized and unbannable. Validating first
        // is also a safety requirement: the candidate records below index
        // `client_ids` and `weights` positionally.
        if weights.is_some() && !matches!(self.method, AggregationMethod::FedAvg) {
            return Err(QoraError::WeightsNotSupported {
                method: format!("{:?}", self.method),
            }
            .into());
        }

        validate_updates(updates)?;
        validate_client_ids(updates, client_ids)?;
        if let Some(w) = weights {
            if w.len() != updates.len() {
                return Err(QoraError::DimensionMismatch.into());
            }
            crate::validation::validate_weights(w)?;
        }
        if let Some(bound) = self.norm_bound {
            // Caller misconfiguration, not client evidence: reported before any
            // update is measured against it.
            validate_norm_bound(bound)?;
        }

        // ---- 2. One decision per submitted update, initially accepted ----
        //
        // Built in submitted order and mutated in place, so the finished list
        // has exactly one entry per input, indices contiguous from zero, and no
        // sorting or repair afterwards.
        let mut decisions: Vec<AggregationAuditDecision> = (0..updates.len())
            .map(|index| {
                AggregationAuditDecision::accepted(index, client_ids.map(|ids| ids[index].clone()))
            })
            .collect();

        let mut candidates: Vec<Candidate<'_>> = (0..updates.len())
            .map(|index| Candidate {
                original_index: index,
                update: &updates[index],
                client_id: client_ids.map(|ids| ids[index].as_str()),
                weight: weights.map(|w| w[index]),
            })
            .collect();

        // ---- 3. Reputation gating ----
        if self.ban_threshold > 0.0 {
            if let Some(ids) = client_ids {
                let threshold = self.ban_threshold;
                let reputation = &self.reputation;
                candidates.retain(|candidate| {
                    let score = reputation.get_score(ids[candidate.original_index].as_str());
                    if score >= threshold {
                        return true;
                    }
                    decisions[candidate.original_index].decision = AggregationDecision::Rejected(
                        AggregationRejectionReason::ReputationBelowThreshold { score, threshold },
                    );
                    false
                });
            }
        }

        // ---- 4. Norm-bound filtering, on the candidates still active ----
        //
        // A client reputation already removed is not measured again: the first
        // policy to reject an update owns its reason, and one update never
        // carries two.
        if let Some(bound) = self.norm_bound {
            let mut retained = Vec::with_capacity(candidates.len());
            for candidate in candidates {
                match evaluate_norm_bound(candidate.update, bound) {
                    Ok(_) => retained.push(candidate),
                    Err(QoraError::NormBoundExceeded { norm, bound }) => {
                        decisions[candidate.original_index].decision =
                            AggregationDecision::Rejected(
                                AggregationRejectionReason::NormBoundExceeded { norm, bound },
                            );
                    }
                    // Unreachable: the batch is validated finite and the bound
                    // is validated above. Reported rather than folded into a
                    // rejection, so that a future change cannot turn malformed
                    // input into a silent exclusion.
                    Err(other) => return Err(other.into()),
                }
            }
            candidates = retained;
        }

        // ---- 5. Fail closed ----
        //
        // Reinstating the rejected clients here would defeat the policy exactly
        // when it distrusts the whole cohort -- the one round where it matters
        // most. Returning before any aggregation also means no score moves, so a
        // caller retrying a rejected round does not compound penalties.
        //
        // Nothing is resolved before this point, so the refused round's record
        // has no effective parameters to report -- which is what it records.
        if candidates.is_empty() {
            return Err(RunFailure {
                error: QoraError::AllUpdatesRejected {
                    submitted: updates.len(),
                },
                rejected: Some(RejectedRound {
                    method: self.refused_round_method(),
                    decisions,
                }),
            });
        }

        // ---- 6. The accepted cohort ----
        let accepted: Vec<Array2<f32>> = candidates
            .iter()
            .map(|candidate| candidate.update.clone())
            .collect();
        let accepted_count = accepted.len();

        // Collecting through `Option` keeps the alignment structural rather than
        // assumed: the weighted vector exists exactly when every surviving
        // candidate carries a weight, which is exactly when the batch supplied
        // them. A rejected client's weight left with its update.
        let accepted_weights: Option<Vec<f32>> = candidates
            .iter()
            .map(|candidate| candidate.weight)
            .collect();

        // ---- 7. Effective parameters and method preconditions ----
        //
        // Resolution happens here, once, and only because a cohort survived to
        // aggregate. Each resolved value is bound to a local that is passed to
        // the aggregation call *and* recorded, so the record cannot name a
        // parameter other than the one that ran.
        //
        // Preconditions are checked against the accepted cohort, never the
        // submitted one. Filtering can shrink a valid cohort into one the method
        // refuses; that is reported rather than repaired.
        let (aggregate, method) = match self.method {
            AggregationMethod::TrimmedMean => {
                let effective_trim_fraction = if self.adaptive_trim {
                    adaptive::compute_adaptive_trim(
                        self.reputation.scores(),
                        0.4,
                        0.05,
                        self.trim_fraction,
                    )
                } else {
                    self.trim_fraction
                };

                (
                    trimmed_mean(&accepted, effective_trim_fraction)?,
                    AuditedAggregationMethod::TrimmedMean {
                        configured_trim_fraction: self.trim_fraction,
                        effective_trim_fraction: Some(effective_trim_fraction),
                        adaptive: self.adaptive_trim,
                    },
                )
            }
            AggregationMethod::Median => (median(&accepted)?, AuditedAggregationMethod::Median),
            AggregationMethod::FedAvg => (
                fedavg(&accepted, accepted_weights.as_deref())?,
                AuditedAggregationMethod::FedAvg,
            ),
            AggregationMethod::Krum(f) => {
                let best_index = aggregate_krum_bfp16(&encode_bfp16(&accepted), f).ok_or(
                    QoraError::InsufficientQuorum {
                        needed: krum_min_clients(f),
                        actual: accepted_count,
                    },
                )?;

                (
                    accepted[best_index].clone(),
                    AuditedAggregationMethod::Krum { f },
                )
            }
            AggregationMethod::MultiKrum(f, requested_m) => {
                // Quorum first: below it there is no safe m at all, and
                // reporting a maximum of 0 would be less actionable than
                // naming the real shortfall.
                if !krum_condition_met(accepted_count, f) {
                    return Err(QoraError::InsufficientQuorum {
                        needed: krum_min_clients(f),
                        actual: accepted_count,
                    }
                    .into());
                }

                let maximum = max_multi_krum_m(accepted_count, f);
                let effective_m = match requested_m {
                    // Omitted: cap to the safe maximum. Preserves the
                    // historical m=3 wherever it is valid, and degrades to
                    // fewer selections for small cohorts rather than failing.
                    None => DEFAULT_MULTI_KRUM_M.min(maximum),
                    // Explicit: honored or refused, never rewritten -- and
                    // never quietly reduced because filtering shrank `n`.
                    Some(requested) if requested >= 1 && requested <= maximum => requested,
                    Some(requested) => {
                        return Err(QoraError::InvalidMultiKrumSelection {
                            clients: accepted_count,
                            byzantine: f,
                            selected: requested,
                            maximum,
                        }
                        .into())
                    }
                };

                let indices = aggregate_multi_krum_bfp16(&encode_bfp16(&accepted), f, effective_m)
                    .ok_or(QoraError::InsufficientQuorum {
                        needed: krum_min_clients(f),
                        actual: accepted_count,
                    })?;

                // Average the selected original f32 vectors. The divisor is the
                // same local the record reports, so the two cannot disagree.
                let mut average = Array2::<f32>::zeros(accepted[0].raw_dim());
                for &index in &indices {
                    average += &accepted[index];
                }
                average /= effective_m as f32;

                (
                    average,
                    AuditedAggregationMethod::MultiKrum {
                        f,
                        requested_m,
                        effective_m: Some(effective_m),
                    },
                )
            }
        };

        // ---- 8. Reputation moves only for clients in the aggregate ----
        //
        // A client filtered out is absent from the distance measurement, which
        // is the whole of its reputation consequence: exclusion is a
        // participation decision, not a verdict, so a norm rejection neither
        // rewards nor penalizes.
        if client_ids.is_some() {
            let accepted_ids: Vec<String> = candidates
                .iter()
                .filter_map(|candidate| candidate.client_id.map(str::to_string))
                .collect();
            self.update_reputations(&accepted_ids, &accepted, &aggregate)?;
        }

        Ok(RunOutcome {
            aggregate,
            method,
            decisions,
        })
    }

    /// Describe the configured method for a round that aggregated nothing.
    ///
    /// **No effective parameters.** Nothing executed, so every runtime-resolved
    /// value is `None` -- there is no trim fraction that was applied and no
    /// Multi-Krum selection count that was used. Inventing one would be a
    /// falsehood in the record: a bare Multi-Krum resolves `m` against the
    /// accepted cohort, and with an empty cohort the uncapped default is not
    /// "what would have run", merely what the configuration starts from.
    ///
    /// Configuration is still reported in full: the trim fraction the aggregator
    /// was built with, the `adaptive` flag, `f`, and the `m` the caller
    /// requested are all known regardless of what ran.
    fn refused_round_method(&self) -> AuditedAggregationMethod {
        match self.method {
            AggregationMethod::TrimmedMean => AuditedAggregationMethod::TrimmedMean {
                configured_trim_fraction: self.trim_fraction,
                effective_trim_fraction: None,
                adaptive: self.adaptive_trim,
            },
            AggregationMethod::Median => AuditedAggregationMethod::Median,
            AggregationMethod::FedAvg => AuditedAggregationMethod::FedAvg,
            AggregationMethod::Krum(f) => AuditedAggregationMethod::Krum { f },
            AggregationMethod::MultiKrum(f, requested_m) => AuditedAggregationMethod::MultiKrum {
                f,
                requested_m,
                effective_m: None,
            },
        }
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
