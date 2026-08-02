//! Aggregation results that carry an audit record.
//!
//! [`crate::ByzantineAggregator::aggregate`] returns only the aggregate, and
//! only the error when a round fails. The audited API in this module returns
//! the same values plus the [`AggregationAuditEntry`] describing what the round
//! decided about each submitted update.
//!
//! Both APIs run the *same* engine. The audited one keeps the record; the
//! ordinary one drops it. Neither filters twice, resolves effective parameters
//! twice, nor can disagree about what happened.
//!
//! # Why the error type is not just `QoraError`
//!
//! A `Result<AuditedAggregation, QoraError>` would discard the audit record in
//! the one case it is most needed: the round where every update was rejected and
//! no aggregate exists. [`AuditedAggregationError`] therefore carries the
//! underlying error *and* the record, when one could be produced.
//!
//! # Records are still caller-owned
//!
//! Nothing here persists, logs, or accumulates. An entry is handed to the caller
//! once and forgotten; [`crate::ByzantineAggregator`] retains no audit state and
//! its serialized form is unchanged by this API.

use std::error::Error;
use std::fmt;

use ndarray::Array2;

use crate::audit::AggregationAuditEntry;
use crate::error::QoraError;

/// A successful aggregation together with its audit record.
///
/// Fields are private and reached through getters so that the struct can gain
/// information later without breaking construction or destructuring at call
/// sites.
#[derive(Clone, Debug)]
pub struct AuditedAggregation {
    aggregate: Array2<f32>,
    audit: AggregationAuditEntry,
}

impl AuditedAggregation {
    /// Build a result pair. Crate-visible: only the aggregation engine is in a
    /// position to produce a record that matches the aggregate beside it.
    pub(crate) fn new(aggregate: Array2<f32>, audit: AggregationAuditEntry) -> Self {
        Self { aggregate, audit }
    }

    /// The aggregated update.
    ///
    /// Identical to what [`crate::ByzantineAggregator::aggregate`] would have
    /// returned for the same inputs and configuration.
    pub fn aggregate(&self) -> &Array2<f32> {
        &self.aggregate
    }

    /// What the round decided about each submitted update.
    ///
    /// The outcome is always [`crate::AggregationAuditOutcome::Aggregated`]:
    /// this type only exists when an aggregate was produced.
    pub fn audit(&self) -> &AggregationAuditEntry {
        &self.audit
    }

    /// Consume the result, yielding the aggregate and the record.
    pub fn into_parts(self) -> (Array2<f32>, AggregationAuditEntry) {
        (self.aggregate, self.audit)
    }
}

/// A failed audited aggregation, with the audit record when one exists.
///
/// # When the record is present
///
/// | Failure | `audit()` |
/// |---|---|
/// | Input validation (empty, dimension mismatch, non-finite value or weight, client-ID count) | `None` |
/// | Invalid configuration (unusable norm bound, weights for a robust method) | `None` |
/// | Every update rejected by filtering | `Some` |
/// | Method precondition failed after filtering (Krum quorum, Multi-Krum `m`) | `None` |
/// | Reputation update failed after aggregating | `None` |
///
/// The last two are a deliberate limitation rather than an oversight. Schema
/// version 1 of [`AggregationAuditEntry`] has exactly two outcomes --
/// aggregated, and everything rejected -- and a round where some candidates
/// survived but the method refused the reduced cohort is neither. Attaching an
/// `Aggregated` entry when no aggregate exists, or an `AllUpdatesRejected` entry
/// when updates were in fact accepted, would make the record describe something
/// that did not happen. A future schema version may add a failure outcome; until
/// then the typed error is returned on its own.
#[derive(Debug)]
pub struct AuditedAggregationError {
    source: QoraError,
    audit: Option<AggregationAuditEntry>,
}

impl AuditedAggregationError {
    /// Build a failure carrying no record.
    pub(crate) fn bare(source: QoraError) -> Self {
        Self {
            source,
            audit: None,
        }
    }

    /// Build a failure carrying the record that explains it.
    pub(crate) fn with_audit(source: QoraError, audit: Option<AggregationAuditEntry>) -> Self {
        Self { source, audit }
    }

    /// The error the ordinary aggregation API would have returned.
    pub fn source_error(&self) -> &QoraError {
        &self.source
    }

    /// The audit record, when the failure is one the schema can describe.
    ///
    /// See the type documentation for exactly which failures carry one.
    pub fn audit(&self) -> Option<&AggregationAuditEntry> {
        self.audit.as_ref()
    }

    /// Consume the failure, yielding the error and the record.
    pub fn into_parts(self) -> (QoraError, Option<AggregationAuditEntry>) {
        (self.source, self.audit)
    }

    /// Discard the record, keeping the error.
    ///
    /// How the ordinary API reports a failure the engine described in full: the
    /// two APIs return the same error for the same round, and only the audited
    /// one keeps the reasons.
    pub(crate) fn into_source(self) -> QoraError {
        self.source
    }
}

impl fmt::Display for AuditedAggregationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The audit record is structured data for a caller to inspect, not
        // something to render into a log line; the message stays the error's.
        write!(f, "{}", self.source)
    }
}

impl Error for AuditedAggregationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

impl From<AuditedAggregationError> for QoraError {
    fn from(error: AuditedAggregationError) -> Self {
        error.into_source()
    }
}
