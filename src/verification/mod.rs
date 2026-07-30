//! Verification primitives for federated learning updates.
//!
//! Provides practical verification checks for model updates:
//!
//! - [`norm_bound`] — Reject updates with excessive L2 norm
//! - [`krum_condition`] — Check Krum's `n >= 2f+3` requirement
//! - [`audit`] — Append-only aggregation audit log

pub mod audit;
pub mod krum_condition;
pub mod norm_bound;

pub use audit::{AggregationAuditEntry, AuditLog};
pub use krum_condition::{krum_condition_met, krum_min_clients, max_multi_krum_m, max_tolerable_f};
pub use norm_bound::check_norm_bound;

// The re-export itself trips the lint; the attribute is scoped to this single
// item so that no other deprecated use can hide behind it. Callers reaching
// `filter_by_norm_bound` through this path still get the deprecation warning.
#[allow(deprecated)]
pub use norm_bound::filter_by_norm_bound;
