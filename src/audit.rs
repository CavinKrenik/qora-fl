//! Aggregation audit schema.
//!
//! An [`AggregationAuditEntry`] records what an aggregation attempt decided
//! about each submitted update: which were accepted, which were rejected and
//! why, and whether an aggregate was produced.
//!
//! # Status
//!
//! **Experimental, and not yet produced by any aggregation path.**
//! [`crate::ByzantineAggregator::aggregate`] does not return audit entries;
//! wiring them in is future work. This module defines the shape so that the
//! serialized contract can be reviewed on its own, before behavior depends on
//! it.
//!
//! # Caller-owned
//!
//! Qora-FL describes what happened; the caller decides whether, where, and how
//! to store it. The library does not write files, keep a log, emit telemetry,
//! choose a retention policy, or retain entries inside
//! [`crate::ByzantineAggregator`]. An entry derives `Serialize`/`Deserialize`
//! and nothing more.
//!
//! This is deliberate. An audit log embedded in the aggregator would grow
//! without bound in the number of rounds and would be dragged into every
//! `to_json` checkpoint, which a long-running server cannot afford.
//!
//! A serializable record is **not** a tamper-proof audit log. Nothing here is
//! signed, hashed, or ordered; integrity is entirely the caller's problem.
//!
//! # What an entry deliberately omits
//!
//! * **No timestamp, round number, or experiment identifier.** The library has
//!   no clock and no notion of a training round. Callers who need those wrap
//!   the entry in their own record:
//!
//!   ```ignore
//!   struct StoredAuditRecord {
//!       timestamp: DateTime<Utc>,
//!       round: u64,
//!       audit: AggregationAuditEntry,
//!   }
//!   ```
//!
//! * **No model data.** No update vectors, aggregate values, coordinates,
//!   layers, or weights. Not even a hash: an unkeyed hash of model data can
//!   still leak information while creating a false impression of
//!   cryptographic auditability.
//!
//! Counts, dispositions, method configuration, and the measurements behind each
//! rejection are what the schema carries.
//!
//! # Privacy
//!
//! [`AggregationAuditDecision::client_id`] carries whatever identifier the
//! caller supplied to the aggregator, which may be sensitive. Since persistence
//! is caller-owned, so is that decision.
//!
//! # Forward compatibility
//!
//! [`AGGREGATION_AUDIT_SCHEMA_VERSION`] versions the serialized shape, not the
//! crate. [`AggregationRejectionReason`] and [`AggregationAuditOutcome`] are
//! `#[non_exhaustive]`: consumers must handle variants they do not recognize
//! and should not assume the current lists are complete. Schema versioning does
//! not make older consumers accept newer variants -- it only lets them detect
//! that they cannot.
//!
//! Wire stability is not promised while the schema is experimental.

use serde::{Deserialize, Serialize};

use crate::error::QoraError;

/// Widest trim fraction `trimmed_mean` accepts; mirrors its own range check.
const MAX_TRIM_FRACTION: f32 = 0.5;

/// Version of the serialized [`AggregationAuditEntry`] shape.
///
/// This versions the schema, **not** the Qora-FL package. A change that alters
/// how an existing field deserializes requires incrementing it; adding an enum
/// variant does not, though it can still break a consumer that matched
/// exhaustively.
///
/// Deserialization rejects any other value: a record written by a newer schema
/// cannot be read correctly by this one, and failing is better than
/// misinterpreting it.
pub const AGGREGATION_AUDIT_SCHEMA_VERSION: u16 = 1;

/// The aggregation method as it actually ran.
///
/// Deliberately *not* [`crate::AggregationMethod`], which records only what was
/// configured. Two of the methods resolve parameters at aggregation time, and a
/// record of the configuration alone cannot explain what happened:
///
/// * With `adaptive_trim`, the trim fraction is recomputed each round from the
///   reputation distribution. `AggregationMethod::TrimmedMean` carries no
///   fraction at all, so an entry naming it would omit the single number that
///   determined how much of each coordinate was discarded.
/// * Bare `"multi_krum"` resolves `m` to `min(3, n - 2f - 2)`. An entry showing
///   `requested_m: None` without the resolved value would not say how many
///   vectors were averaged.
///
/// Both the configured intent and the executed value are kept, so a reader can
/// tell a deliberate setting from a runtime resolution.
///
/// Using an enum rather than a flat struct makes impossible combinations
/// unrepresentable -- a FedAvg record cannot carry a trim fraction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AuditedAggregationMethod {
    /// Standard FedAvg. No runtime-resolved parameters.
    ///
    /// Renamed explicitly: `snake_case` would yield `fed_avg`, while every
    /// other surface in the project -- the Python method string, the docs --
    /// spells it `fedavg`.
    #[serde(rename = "fedavg")]
    FedAvg,
    /// Coordinate-wise median. No runtime-resolved parameters.
    Median,
    /// Coordinate-wise trimmed mean.
    TrimmedMean {
        /// The fraction the aggregator was constructed with.
        configured_trim_fraction: f32,
        /// The fraction actually applied this round.
        ///
        /// Equal to `configured_trim_fraction` unless `adaptive` is true.
        effective_trim_fraction: f32,
        /// Whether the effective fraction was computed from reputation.
        adaptive: bool,
    },
    /// Krum.
    Krum {
        /// Configured upper bound on Byzantine clients.
        f: usize,
    },
    /// Multi-Krum.
    MultiKrum {
        /// Configured upper bound on Byzantine clients.
        f: usize,
        /// The `m` the caller asked for, or `None` for the bare form.
        requested_m: Option<usize>,
        /// The number of vectors actually selected and averaged.
        effective_m: usize,
    },
}

/// Why a single update was excluded from aggregation.
///
/// Covers per-update *filtering* decisions only. Whole-request failures --
/// dimension mismatch, non-finite values, invalid weights, Krum quorum,
/// Multi-Krum configuration -- abort before any update is individually judged,
/// so they are not rejection reasons; they would belong to a future
/// outcome-level failure representation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AggregationRejectionReason {
    /// The client's reputation was below the configured ban threshold.
    ReputationBelowThreshold {
        /// The client's score at the time of the decision, within `[0, 1]`.
        score: f32,
        /// The configured ban threshold, within `[0, 1]`.
        threshold: f32,
    },
    /// The update's L2 norm exceeded the configured bound.
    ///
    /// `norm` is `f64` because [`crate::verification::check_norm_bound`]
    /// computes it in `f64`; narrowing it here would discard the precision
    /// that check exists to preserve.
    NormBoundExceeded {
        /// The update's computed L2 norm.
        norm: f64,
        /// The configured bound it exceeded.
        bound: f32,
    },
}

/// The disposition of one submitted update.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregationDecision {
    /// The update participated in aggregation.
    Accepted,
    /// The update was filtered out, for the given reason.
    Rejected(AggregationRejectionReason),
}

impl AggregationDecision {
    /// Whether this decision accepted the update.
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted)
    }

    /// The rejection reason, or `None` if the update was accepted.
    pub fn rejection_reason(&self) -> Option<&AggregationRejectionReason> {
        match self {
            Self::Accepted => None,
            Self::Rejected(reason) => Some(reason),
        }
    }
}

/// What happened to one submitted update.
///
/// One record per submitted update, rather than separate accepted and rejected
/// lists: every input then has exactly one explicit disposition, an update
/// cannot appear in both lists or neither, and the counts are derived rather
/// than stored and kept in sync.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AggregationAuditDecision {
    /// Position of the update in the caller's original slice.
    ///
    /// Always the **submitted** position. Filtering does not renumber:
    /// rejecting update 1 of 4 leaves the remaining indices as 0, 2, 3, so a
    /// caller can map every decision back to what they sent.
    pub update_index: usize,
    /// The identifier the caller supplied, if any.
    ///
    /// `None` when aggregation ran without `client_ids`. Never synthesized
    /// from the position -- absence of an identity is recorded as absence.
    pub client_id: Option<String>,
    /// Whether the update was accepted or rejected.
    pub decision: AggregationDecision,
}

impl AggregationAuditDecision {
    /// Record an accepted update.
    pub fn accepted(update_index: usize, client_id: Option<String>) -> Self {
        Self {
            update_index,
            client_id,
            decision: AggregationDecision::Accepted,
        }
    }

    /// Record a rejected update.
    pub fn rejected(
        update_index: usize,
        client_id: Option<String>,
        reason: AggregationRejectionReason,
    ) -> Self {
        Self {
            update_index,
            client_id,
            decision: AggregationDecision::Rejected(reason),
        }
    }
}

/// Whether the attempt produced an aggregate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AggregationAuditOutcome {
    /// An aggregate was produced from the accepted updates.
    Aggregated,
    /// Every submitted update was rejected, so no aggregate exists.
    ///
    /// Recorded rather than omitted: a round that produced nothing is the one
    /// most worth explaining, and the decisions say which policy excluded each
    /// client.
    AllUpdatesRejected,
}

/// A record of one aggregation attempt.
///
/// See the [module documentation](self) for status, ownership, and what the
/// schema deliberately omits.
///
/// # Invariants
///
/// Enforced on construction *and* on deserialization, so a record cannot
/// describe an impossible attempt:
///
/// * At least one decision.
/// * Decision indices are unique and contiguous from 0 -- one per submitted
///   update, in submitted order.
/// * [`AggregationAuditOutcome::Aggregated`] has at least one accepted update.
/// * [`AggregationAuditOutcome::AllUpdatesRejected`] has none.
/// * Reputation scores and thresholds are finite and within `[0, 1]`.
/// * Norms are finite and non-negative; bounds are finite and positive,
///   matching what [`crate::verification::check_norm_bound`] accepts.
/// * `schema_version` equals [`AGGREGATION_AUDIT_SCHEMA_VERSION`].
/// * Trim fractions are finite and within `0.0..=0.5`, and a non-adaptive
///   record has matching configured and effective fractions.
/// * `f` is small enough that `2f + 3` does not overflow.
/// * Multi-Krum's `effective_m` is at least 1, and an explicit `requested_m`
///   equals it -- an explicit request is honored exactly or refused, never
///   silently resolved to something else.
///
/// `effective_m` is deliberately **not** checked against the accepted cohort
/// size: the entry does not know `n` at the moment parameters were resolved,
/// so that relationship belongs to the integration that builds the record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "AuditEntryRepr")]
pub struct AggregationAuditEntry {
    schema_version: u16,
    method: AuditedAggregationMethod,
    decisions: Vec<AggregationAuditDecision>,
    outcome: AggregationAuditOutcome,
}

/// Deserialization shadow, so the invariants above run before an entry exists.
#[derive(Deserialize)]
struct AuditEntryRepr {
    schema_version: u16,
    method: AuditedAggregationMethod,
    decisions: Vec<AggregationAuditDecision>,
    outcome: AggregationAuditOutcome,
}

impl TryFrom<AuditEntryRepr> for AggregationAuditEntry {
    type Error = QoraError;

    fn try_from(repr: AuditEntryRepr) -> Result<Self, Self::Error> {
        if repr.schema_version != AGGREGATION_AUDIT_SCHEMA_VERSION {
            return Err(QoraError::InvalidAuditEntry {
                reason: format!(
                    "unsupported audit schema version {} (this build reads {})",
                    repr.schema_version, AGGREGATION_AUDIT_SCHEMA_VERSION
                ),
            });
        }
        Self::new(repr.method, repr.decisions, repr.outcome)
    }
}

impl AggregationAuditEntry {
    /// Build a validated entry, stamped with the current schema version.
    ///
    /// Crate-visible: entries describe what an aggregation path decided, and
    /// no such path produces them yet. External code obtains entries by
    /// deserializing them, which enforces the same invariants.
    ///
    /// # Errors
    ///
    /// [`QoraError::InvalidAuditEntry`] if any invariant listed on the type is
    /// violated.
    pub(crate) fn new(
        method: AuditedAggregationMethod,
        decisions: Vec<AggregationAuditDecision>,
        outcome: AggregationAuditOutcome,
    ) -> Result<Self, QoraError> {
        let invalid = |reason: String| QoraError::InvalidAuditEntry { reason };

        validate_method(&method).map_err(invalid)?;

        if decisions.is_empty() {
            return Err(invalid("an audit entry needs at least one decision".into()));
        }

        // Unique and contiguous from 0: exactly one decision per submitted
        // update, in submitted order.
        let mut seen = vec![false; decisions.len()];
        for decision in &decisions {
            match seen.get_mut(decision.update_index) {
                None => {
                    return Err(invalid(format!(
                        "update_index {} is out of range for {} decisions",
                        decision.update_index,
                        decisions.len()
                    )))
                }
                Some(true) => {
                    return Err(invalid(format!(
                        "duplicate update_index {}",
                        decision.update_index
                    )))
                }
                Some(slot) => *slot = true,
            }

            if let Some(reason) = decision.decision.rejection_reason() {
                validate_reason(reason).map_err(invalid)?;
            }
        }

        let accepted = decisions
            .iter()
            .filter(|d| d.decision.is_accepted())
            .count();
        match outcome {
            AggregationAuditOutcome::Aggregated if accepted == 0 => {
                return Err(invalid(
                    "outcome is Aggregated but no update was accepted".into(),
                ))
            }
            AggregationAuditOutcome::AllUpdatesRejected if accepted > 0 => {
                return Err(invalid(format!(
                    "outcome is AllUpdatesRejected but {} update(s) were accepted",
                    accepted
                )))
            }
            _ => {}
        }

        Ok(Self {
            schema_version: AGGREGATION_AUDIT_SCHEMA_VERSION,
            method,
            decisions,
            outcome,
        })
    }

    /// Schema version this entry was written with.
    pub fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// The aggregation method as it actually ran.
    ///
    /// Carries the effective parameters, not just the configuration: the trim
    /// fraction actually applied and the Multi-Krum count actually selected,
    /// alongside what was requested.
    pub fn method(&self) -> &AuditedAggregationMethod {
        &self.method
    }

    /// One decision per submitted update, in submitted order.
    pub fn decisions(&self) -> &[AggregationAuditDecision] {
        &self.decisions
    }

    /// Whether an aggregate was produced.
    pub fn outcome(&self) -> &AggregationAuditOutcome {
        &self.outcome
    }

    /// Number of updates submitted.
    ///
    /// Derived from the decisions rather than stored, so it cannot disagree
    /// with them.
    pub fn submitted_count(&self) -> usize {
        self.decisions.len()
    }

    /// Number of updates that participated in aggregation.
    pub fn accepted_count(&self) -> usize {
        self.decisions
            .iter()
            .filter(|d| d.decision.is_accepted())
            .count()
    }

    /// Number of updates filtered out.
    pub fn rejected_count(&self) -> usize {
        self.submitted_count() - self.accepted_count()
    }
}

/// Check the parameters carried by an effective-method descriptor.
fn validate_method(method: &AuditedAggregationMethod) -> Result<(), String> {
    // `2f + 3` is the Krum quorum requirement; an `f` that overflows it cannot
    // describe a real configuration.
    let representable = |f: usize| f.checked_mul(2).and_then(|x| x.checked_add(3)).is_some();

    match method {
        AuditedAggregationMethod::FedAvg | AuditedAggregationMethod::Median => Ok(()),

        AuditedAggregationMethod::TrimmedMean {
            configured_trim_fraction,
            effective_trim_fraction,
            adaptive,
        } => {
            for (label, value) in [
                ("configured", configured_trim_fraction),
                ("effective", effective_trim_fraction),
            ] {
                if !value.is_finite() || !(0.0..=MAX_TRIM_FRACTION).contains(value) {
                    return Err(format!(
                        "{} trim fraction {} is not a finite value in [0, {}]",
                        label, value, MAX_TRIM_FRACTION
                    ));
                }
            }
            if !adaptive && configured_trim_fraction != effective_trim_fraction {
                return Err(format!(
                    "non-adaptive trimmed mean must apply its configured fraction,                      but configured {} != effective {}",
                    configured_trim_fraction, effective_trim_fraction
                ));
            }
            Ok(())
        }

        AuditedAggregationMethod::Krum { f } => {
            if !representable(*f) {
                return Err(format!("Krum f {} overflows the quorum requirement", f));
            }
            Ok(())
        }

        AuditedAggregationMethod::MultiKrum {
            f,
            requested_m,
            effective_m,
        } => {
            if !representable(*f) {
                return Err(format!(
                    "Multi-Krum f {} overflows the quorum requirement",
                    f
                ));
            }
            if *effective_m < 1 {
                return Err("Multi-Krum must select at least one vector".to_string());
            }
            if let Some(requested) = requested_m {
                if requested != effective_m {
                    return Err(format!(
                        "explicit Multi-Krum m {} was recorded as effective {};                          an explicit request is honored exactly or refused",
                        requested, effective_m
                    ));
                }
            }
            Ok(())
        }
    }
}

/// Check the measurements carried by a rejection reason.
fn validate_reason(reason: &AggregationRejectionReason) -> Result<(), String> {
    match reason {
        AggregationRejectionReason::ReputationBelowThreshold { score, threshold } => {
            for (label, value) in [("score", score), ("threshold", threshold)] {
                if !value.is_finite() || !(0.0..=1.0).contains(value) {
                    return Err(format!(
                        "reputation {} {} is not a finite value in [0, 1]",
                        label, value
                    ));
                }
            }
            Ok(())
        }
        AggregationRejectionReason::NormBoundExceeded { norm, bound } => {
            if !norm.is_finite() || *norm < 0.0 {
                return Err(format!("norm {} is not finite and non-negative", norm));
            }
            // Mirrors check_norm_bound: a bound must be finite and > 0.
            if !bound.is_finite() || *bound <= 0.0 {
                return Err(format!("norm bound {} is not finite and positive", bound));
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Static trimmed mean: configured and effective fractions agree.
    fn static_trim() -> AuditedAggregationMethod {
        AuditedAggregationMethod::TrimmedMean {
            configured_trim_fraction: 0.25,
            effective_trim_fraction: 0.25,
            adaptive: false,
        }
    }

    fn accepted(i: usize, id: &str) -> AggregationAuditDecision {
        AggregationAuditDecision::accepted(i, Some(id.to_string()))
    }

    fn rep_rejected(i: usize, id: &str) -> AggregationAuditDecision {
        AggregationAuditDecision::rejected(
            i,
            Some(id.to_string()),
            AggregationRejectionReason::ReputationBelowThreshold {
                score: 0.1,
                threshold: 0.2,
            },
        )
    }

    fn norm_rejected(i: usize, id: Option<&str>) -> AggregationAuditDecision {
        AggregationAuditDecision::rejected(
            i,
            id.map(str::to_string),
            AggregationRejectionReason::NormBoundExceeded {
                norm: 1.414_213_5e20,
                bound: 10.0,
            },
        )
    }

    // ===== Construction =====

    #[test]
    fn accepts_a_fully_accepted_entry() {
        let entry = AggregationAuditEntry::new(
            AuditedAggregationMethod::Median,
            vec![accepted(0, "a"), accepted(1, "b")],
            AggregationAuditOutcome::Aggregated,
        )
        .expect("valid entry");

        assert_eq!(entry.schema_version(), AGGREGATION_AUDIT_SCHEMA_VERSION);
        assert_eq!(entry.submitted_count(), 2);
        assert_eq!(entry.accepted_count(), 2);
        assert_eq!(entry.rejected_count(), 0);
    }

    #[test]
    fn accepts_a_mixed_entry() {
        let entry = AggregationAuditEntry::new(
            static_trim(),
            vec![accepted(0, "a"), rep_rejected(1, "b"), accepted(2, "c")],
            AggregationAuditOutcome::Aggregated,
        )
        .expect("valid entry");

        assert_eq!(entry.submitted_count(), 3);
        assert_eq!(entry.accepted_count(), 2);
        assert_eq!(entry.rejected_count(), 1);
    }

    #[test]
    fn accepts_an_all_rejected_entry() {
        let entry = AggregationAuditEntry::new(
            AuditedAggregationMethod::FedAvg,
            vec![rep_rejected(0, "a"), norm_rejected(1, Some("b"))],
            AggregationAuditOutcome::AllUpdatesRejected,
        )
        .expect("valid entry");

        assert_eq!(entry.accepted_count(), 0);
        assert_eq!(entry.rejected_count(), 2);
        assert_eq!(
            entry.outcome(),
            &AggregationAuditOutcome::AllUpdatesRejected
        );
    }

    #[test]
    fn accepts_entries_without_client_ids() {
        let entry = AggregationAuditEntry::new(
            AuditedAggregationMethod::Median,
            vec![
                AggregationAuditDecision::accepted(0, None),
                norm_rejected(1, None),
                AggregationAuditDecision::accepted(2, None),
            ],
            AggregationAuditOutcome::Aggregated,
        )
        .expect("positions alone are sufficient");

        assert!(entry.decisions().iter().all(|d| d.client_id.is_none()));
        assert_eq!(entry.rejected_count(), 1);
    }

    // ===== Original positions =====

    #[test]
    fn rejection_does_not_renumber_the_survivors() {
        // Update 1 of 4 is rejected; the rest keep their submitted positions.
        let entry = AggregationAuditEntry::new(
            AuditedAggregationMethod::Median,
            vec![
                accepted(0, "a"),
                rep_rejected(1, "b"),
                accepted(2, "c"),
                accepted(3, "d"),
            ],
            AggregationAuditOutcome::Aggregated,
        )
        .unwrap();

        let indices: Vec<usize> = entry.decisions().iter().map(|d| d.update_index).collect();
        assert_eq!(indices, vec![0, 1, 2, 3]);

        let accepted_indices: Vec<usize> = entry
            .decisions()
            .iter()
            .filter(|d| d.decision.is_accepted())
            .map(|d| d.update_index)
            .collect();
        assert_eq!(accepted_indices, vec![0, 2, 3]);
    }

    // ===== Invariants =====

    #[test]
    fn rejects_an_empty_decision_list() {
        assert!(AggregationAuditEntry::new(
            AuditedAggregationMethod::Median,
            vec![],
            AggregationAuditOutcome::Aggregated
        )
        .is_err());
    }

    #[test]
    fn rejects_duplicate_indices() {
        assert!(AggregationAuditEntry::new(
            AuditedAggregationMethod::Median,
            vec![accepted(0, "a"), accepted(0, "b")],
            AggregationAuditOutcome::Aggregated
        )
        .is_err());
    }

    #[test]
    fn rejects_non_contiguous_indices() {
        // 0 and 2 for two decisions: update 1 has no disposition.
        assert!(AggregationAuditEntry::new(
            AuditedAggregationMethod::Median,
            vec![accepted(0, "a"), accepted(2, "c")],
            AggregationAuditOutcome::Aggregated
        )
        .is_err());
    }

    #[test]
    fn rejects_aggregated_with_no_accepted_updates() {
        assert!(AggregationAuditEntry::new(
            AuditedAggregationMethod::Median,
            vec![rep_rejected(0, "a")],
            AggregationAuditOutcome::Aggregated
        )
        .is_err());
    }

    #[test]
    fn rejects_all_updates_rejected_with_an_accepted_update() {
        assert!(AggregationAuditEntry::new(
            AuditedAggregationMethod::Median,
            vec![accepted(0, "a"), rep_rejected(1, "b")],
            AggregationAuditOutcome::AllUpdatesRejected
        )
        .is_err());
    }

    #[test]
    fn rejects_out_of_range_reputation_measurements() {
        for (score, threshold) in [
            (f32::NAN, 0.2),
            (0.1, f32::NAN),
            (-0.1, 0.2),
            (1.1, 0.2),
            (0.1, 1.5),
        ] {
            let decision = AggregationAuditDecision::rejected(
                0,
                None,
                AggregationRejectionReason::ReputationBelowThreshold { score, threshold },
            );
            assert!(
                AggregationAuditEntry::new(
                    AuditedAggregationMethod::Median,
                    vec![decision],
                    AggregationAuditOutcome::AllUpdatesRejected
                )
                .is_err(),
                "score={} threshold={} must be rejected",
                score,
                threshold
            );
        }
    }

    #[test]
    fn rejects_invalid_norm_measurements() {
        for (norm, bound) in [
            (f64::NAN, 1.0),
            (f64::INFINITY, 1.0),
            (-1.0, 1.0),
            (1.0, f32::NAN),
            (1.0, 0.0),
            (1.0, -1.0),
        ] {
            let decision = AggregationAuditDecision::rejected(
                0,
                None,
                AggregationRejectionReason::NormBoundExceeded { norm, bound },
            );
            assert!(
                AggregationAuditEntry::new(
                    AuditedAggregationMethod::Median,
                    vec![decision],
                    AggregationAuditOutcome::AllUpdatesRejected
                )
                .is_err(),
                "norm={} bound={} must be rejected",
                norm,
                bound
            );
        }
    }

    // ===== Effective method parameters =====

    #[test]
    fn static_trim_records_identical_configured_and_effective_fractions() {
        let entry = AggregationAuditEntry::new(
            static_trim(),
            vec![accepted(0, "a")],
            AggregationAuditOutcome::Aggregated,
        )
        .unwrap();

        match entry.method() {
            AuditedAggregationMethod::TrimmedMean {
                configured_trim_fraction,
                effective_trim_fraction,
                adaptive,
            } => {
                assert_eq!(configured_trim_fraction, effective_trim_fraction);
                assert!(!adaptive);
            }
            other => panic!("expected trimmed mean, got {:?}", other),
        }
    }

    #[test]
    fn adaptive_trim_records_both_the_baseline_and_the_round_value() {
        // The point of the descriptor: `AggregationMethod::TrimmedMean` alone
        // could not say which fraction was actually applied.
        let entry = AggregationAuditEntry::new(
            AuditedAggregationMethod::TrimmedMean {
                configured_trim_fraction: 0.125,
                effective_trim_fraction: 0.375,
                adaptive: true,
            },
            vec![accepted(0, "a")],
            AggregationAuditOutcome::Aggregated,
        )
        .expect("adaptive fractions may differ");

        match entry.method() {
            AuditedAggregationMethod::TrimmedMean {
                configured_trim_fraction,
                effective_trim_fraction,
                adaptive,
            } => {
                assert_eq!(*configured_trim_fraction, 0.125);
                assert_eq!(*effective_trim_fraction, 0.375);
                assert!(adaptive);
            }
            other => panic!("expected trimmed mean, got {:?}", other),
        }
    }

    #[test]
    fn non_adaptive_trim_must_apply_its_configured_fraction() {
        assert!(AggregationAuditEntry::new(
            AuditedAggregationMethod::TrimmedMean {
                configured_trim_fraction: 0.1,
                effective_trim_fraction: 0.4,
                adaptive: false,
            },
            vec![accepted(0, "a")],
            AggregationAuditOutcome::Aggregated,
        )
        .is_err());
    }

    #[test]
    fn trim_fractions_must_be_finite_and_in_range() {
        for (configured, effective) in [
            (f32::NAN, f32::NAN),
            (f32::INFINITY, f32::INFINITY),
            (-0.1, -0.1),
            (0.6, 0.6),
            (0.25, 0.9),
        ] {
            assert!(
                AggregationAuditEntry::new(
                    AuditedAggregationMethod::TrimmedMean {
                        configured_trim_fraction: configured,
                        effective_trim_fraction: effective,
                        adaptive: true,
                    },
                    vec![accepted(0, "a")],
                    AggregationAuditOutcome::Aggregated,
                )
                .is_err(),
                "configured={} effective={} must be rejected",
                configured,
                effective
            );
        }
    }

    #[test]
    fn krum_preserves_its_byzantine_bound() {
        let entry = AggregationAuditEntry::new(
            AuditedAggregationMethod::Krum { f: 3 },
            vec![accepted(0, "a")],
            AggregationAuditOutcome::Aggregated,
        )
        .unwrap();
        assert_eq!(entry.method(), &AuditedAggregationMethod::Krum { f: 3 });
    }

    #[test]
    fn an_f_that_overflows_the_quorum_requirement_is_rejected() {
        for method in [
            AuditedAggregationMethod::Krum { f: usize::MAX },
            AuditedAggregationMethod::MultiKrum {
                f: usize::MAX,
                requested_m: None,
                effective_m: 1,
            },
        ] {
            assert!(AggregationAuditEntry::new(
                method,
                vec![accepted(0, "a")],
                AggregationAuditOutcome::Aggregated,
            )
            .is_err());
        }
    }

    #[test]
    fn bare_multi_krum_records_its_resolved_selection_count() {
        // requested_m: None with a resolved effective_m -- the whole reason the
        // descriptor keeps both.
        let entry = AggregationAuditEntry::new(
            AuditedAggregationMethod::MultiKrum {
                f: 1,
                requested_m: None,
                effective_m: 2,
            },
            vec![accepted(0, "a")],
            AggregationAuditOutcome::Aggregated,
        )
        .expect("a bare request may resolve to any valid m");

        match entry.method() {
            AuditedAggregationMethod::MultiKrum {
                requested_m,
                effective_m,
                ..
            } => {
                assert_eq!(*requested_m, None);
                assert_eq!(*effective_m, 2);
            }
            other => panic!("expected multi-krum, got {:?}", other),
        }
    }

    #[test]
    fn explicit_multi_krum_must_match_its_effective_count() {
        // An explicit m is honored exactly or refused; a record showing it
        // silently resolved to something else describes behavior that cannot
        // happen.
        assert!(AggregationAuditEntry::new(
            AuditedAggregationMethod::MultiKrum {
                f: 1,
                requested_m: Some(3),
                effective_m: 1,
            },
            vec![accepted(0, "a")],
            AggregationAuditOutcome::Aggregated,
        )
        .is_err());

        assert!(AggregationAuditEntry::new(
            AuditedAggregationMethod::MultiKrum {
                f: 1,
                requested_m: Some(3),
                effective_m: 3,
            },
            vec![accepted(0, "a")],
            AggregationAuditOutcome::Aggregated,
        )
        .is_ok());
    }

    #[test]
    fn multi_krum_must_select_at_least_one_vector() {
        assert!(AggregationAuditEntry::new(
            AuditedAggregationMethod::MultiKrum {
                f: 1,
                requested_m: None,
                effective_m: 0,
            },
            vec![accepted(0, "a")],
            AggregationAuditOutcome::Aggregated,
        )
        .is_err());
    }

    #[test]
    fn deserialization_enforces_the_method_invariants() {
        // Non-adaptive with mismatched fractions.
        let json = r#"{
            "schema_version": 1,
            "method": {"trimmed_mean": {"configured_trim_fraction": 0.1,
                       "effective_trim_fraction": 0.4, "adaptive": false}},
            "decisions": [{"update_index": 0, "client_id": null, "decision": "accepted"}],
            "outcome": "aggregated"
        }"#;
        assert!(serde_json::from_str::<AggregationAuditEntry>(json).is_err());

        // Out-of-range trim fraction.
        let json = r#"{
            "schema_version": 1,
            "method": {"trimmed_mean": {"configured_trim_fraction": 0.9,
                       "effective_trim_fraction": 0.9, "adaptive": false}},
            "decisions": [{"update_index": 0, "client_id": null, "decision": "accepted"}],
            "outcome": "aggregated"
        }"#;
        assert!(serde_json::from_str::<AggregationAuditEntry>(json).is_err());

        // Explicit m disagreeing with the effective count.
        let json = r#"{
            "schema_version": 1,
            "method": {"multi_krum": {"f": 1, "requested_m": 3, "effective_m": 1}},
            "decisions": [{"update_index": 0, "client_id": null, "decision": "accepted"}],
            "outcome": "aggregated"
        }"#;
        assert!(serde_json::from_str::<AggregationAuditEntry>(json).is_err());
    }

    #[test]
    fn serialized_method_names_match_the_project_vocabulary() {
        let name = |method: AuditedAggregationMethod| {
            let entry = AggregationAuditEntry::new(
                method,
                vec![accepted(0, "a")],
                AggregationAuditOutcome::Aggregated,
            )
            .unwrap();
            let value = serde_json::to_value(&entry).unwrap();
            match &value["method"] {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Object(map) => map.keys().next().unwrap().clone(),
                other => panic!("unexpected method encoding {:?}", other),
            }
        };

        // Same spellings the Python method strings use.
        assert_eq!(name(AuditedAggregationMethod::FedAvg), "fedavg");
        assert_eq!(name(AuditedAggregationMethod::Median), "median");
        assert_eq!(name(static_trim()), "trimmed_mean");
        assert_eq!(name(AuditedAggregationMethod::Krum { f: 1 }), "krum");
        assert_eq!(
            name(AuditedAggregationMethod::MultiKrum {
                f: 1,
                requested_m: None,
                effective_m: 1,
            }),
            "multi_krum"
        );
    }

    // ===== Method round-trip =====

    #[test]
    fn every_method_survives_a_round_trip() {
        for method in [
            AuditedAggregationMethod::FedAvg,
            AuditedAggregationMethod::Median,
            static_trim(),
            // Adaptive: the two fractions differ, and both must survive.
            AuditedAggregationMethod::TrimmedMean {
                configured_trim_fraction: 0.125,
                effective_trim_fraction: 0.375,
                adaptive: true,
            },
            AuditedAggregationMethod::Krum { f: 2 },
            // Bare and explicit Multi-Krum must stay distinguishable: they mean
            // different things at the aggregation boundary.
            AuditedAggregationMethod::MultiKrum {
                f: 1,
                requested_m: None,
                effective_m: 2,
            },
            AuditedAggregationMethod::MultiKrum {
                f: 1,
                requested_m: Some(3),
                effective_m: 3,
            },
        ] {
            let entry = AggregationAuditEntry::new(
                method.clone(),
                vec![accepted(0, "a")],
                AggregationAuditOutcome::Aggregated,
            )
            .unwrap();

            let json = serde_json::to_string(&entry).unwrap();
            let restored: AggregationAuditEntry = serde_json::from_str(&json).unwrap();
            assert_eq!(restored.method(), &method);
            assert_eq!(restored, entry);
        }
    }

    // ===== Rejection reasons round-trip =====

    #[test]
    fn rejection_measurements_survive_a_round_trip() {
        let entry = AggregationAuditEntry::new(
            AuditedAggregationMethod::Median,
            vec![
                AggregationAuditDecision::rejected(
                    0,
                    Some("a".into()),
                    AggregationRejectionReason::ReputationBelowThreshold {
                        score: 0.125,
                        threshold: 0.25,
                    },
                ),
                AggregationAuditDecision::rejected(
                    1,
                    Some("b".into()),
                    AggregationRejectionReason::NormBoundExceeded {
                        norm: 1.414_213_562_373e20,
                        bound: 1.5e-3,
                    },
                ),
            ],
            AggregationAuditOutcome::AllUpdatesRejected,
        )
        .unwrap();

        let restored: AggregationAuditEntry =
            serde_json::from_str(&serde_json::to_string(&entry).unwrap()).unwrap();

        match restored.decisions()[0].decision.rejection_reason().unwrap() {
            AggregationRejectionReason::ReputationBelowThreshold { score, threshold } => {
                assert_eq!(*score, 0.125);
                assert_eq!(*threshold, 0.25);
            }
            other => panic!("expected a reputation rejection, got {:?}", other),
        }
        match restored.decisions()[1].decision.rejection_reason().unwrap() {
            AggregationRejectionReason::NormBoundExceeded { norm, bound } => {
                // f64 precision must survive; this value is not f32-representable.
                assert_eq!(*norm, 1.414_213_562_373e20);
                assert_eq!(*bound, 1.5e-3);
            }
            other => panic!("expected a norm rejection, got {:?}", other),
        }
    }

    // ===== Serialization shape =====

    /// Pins the wire format against an explicit fixture, so an accidental
    /// rename or restructure cannot pass unnoticed.
    #[test]
    fn serialized_shape_matches_the_documented_fixture() {
        let entry = AggregationAuditEntry::new(
            static_trim(),
            vec![
                accepted(0, "client-a"),
                AggregationAuditDecision::rejected(
                    1,
                    Some("client-b".into()),
                    // Exactly representable in f32, so the JSON fixture below
                    // is not disturbed by f32 -> f64 widening.
                    AggregationRejectionReason::ReputationBelowThreshold {
                        score: 0.125,
                        threshold: 0.25,
                    },
                ),
                AggregationAuditDecision::accepted(2, None),
            ],
            AggregationAuditOutcome::Aggregated,
        )
        .unwrap();

        let expected = serde_json::json!({
            "schema_version": 1,
            "method": {
                "trimmed_mean": {
                    "configured_trim_fraction": 0.25,
                    "effective_trim_fraction": 0.25,
                    "adaptive": false
                }
            },
            "decisions": [
                {
                    "update_index": 0,
                    "client_id": "client-a",
                    "decision": "accepted"
                },
                {
                    "update_index": 1,
                    "client_id": "client-b",
                    "decision": {
                        "rejected": {
                            "reputation_below_threshold": {
                                "score": 0.125,
                                "threshold": 0.25
                            }
                        }
                    }
                },
                {
                    "update_index": 2,
                    "client_id": null,
                    "decision": "accepted"
                }
            ],
            "outcome": "aggregated"
        });

        assert_eq!(serde_json::to_value(&entry).unwrap(), expected);
    }

    #[test]
    fn norm_rejection_shape_matches_the_documented_fixture() {
        let entry = AggregationAuditEntry::new(
            AuditedAggregationMethod::MultiKrum {
                f: 1,
                requested_m: Some(3),
                effective_m: 3,
            },
            vec![norm_rejected(0, None)],
            AggregationAuditOutcome::AllUpdatesRejected,
        )
        .unwrap();

        let value = serde_json::to_value(&entry).unwrap();
        assert_eq!(
            value["method"],
            serde_json::json!({
                "multi_krum": { "f": 1, "requested_m": 3, "effective_m": 3 }
            })
        );
        assert_eq!(value["outcome"], "all_updates_rejected");
        assert_eq!(
            value["decisions"][0]["decision"]["rejected"]["norm_bound_exceeded"]["bound"],
            serde_json::json!(10.0)
        );
    }

    // ===== Deserialization enforces the same invariants =====

    #[test]
    fn deserialization_rejects_an_impossible_entry() {
        // Outcome says nothing aggregated, yet an update was accepted.
        let json = r#"{
            "schema_version": 1,
            "method": "median",
            "decisions": [
                {"update_index": 0, "client_id": null, "decision": "accepted"}
            ],
            "outcome": "all_updates_rejected"
        }"#;
        assert!(serde_json::from_str::<AggregationAuditEntry>(json).is_err());
    }

    #[test]
    fn deserialization_rejects_duplicate_indices() {
        let json = r#"{
            "schema_version": 1,
            "method": "median",
            "decisions": [
                {"update_index": 0, "client_id": null, "decision": "accepted"},
                {"update_index": 0, "client_id": null, "decision": "accepted"}
            ],
            "outcome": "aggregated"
        }"#;
        assert!(serde_json::from_str::<AggregationAuditEntry>(json).is_err());
    }

    #[test]
    fn deserialization_rejects_an_unsupported_schema_version() {
        let json = r#"{
            "schema_version": 2,
            "method": "median",
            "decisions": [
                {"update_index": 0, "client_id": null, "decision": "accepted"}
            ],
            "outcome": "aggregated"
        }"#;
        assert!(serde_json::from_str::<AggregationAuditEntry>(json).is_err());
    }

    #[test]
    fn deserialization_tolerates_unknown_fields() {
        // Consumers should tolerate additional fields from a future writer.
        let json = r#"{
            "schema_version": 1,
            "method": "median",
            "decisions": [
                {"update_index": 0, "client_id": null, "decision": "accepted"}
            ],
            "outcome": "aggregated",
            "something_added_later": 42
        }"#;
        assert!(serde_json::from_str::<AggregationAuditEntry>(json).is_ok());
    }
}
