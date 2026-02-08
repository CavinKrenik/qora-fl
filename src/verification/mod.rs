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
pub use krum_condition::{krum_condition_met, max_tolerable_f};
pub use norm_bound::{check_norm_bound, filter_by_norm_bound};
