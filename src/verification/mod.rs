//! Verification primitives for federated learning updates.
//!
//! Provides verification primitives for model updates:
//!
//! - [`krum_condition`] — Krum's `n >= 2f+3` requirement and the Multi-Krum
//!   selection bound. **Wired into aggregation** and enforced.
//! - [`norm_bound`] — L2 norm checking. [`norm_bound::check_norm_bound`] is the
//!   checked API: it validates finiteness and the bound itself, and computes
//!   the norm in `f64`. It is **not** invoked by any aggregation path;
//!   integrating it is roadmap work.
//!   [`norm_bound::filter_by_norm_bound`] is deprecated because it discards
//!   verification errors silently.
//! - [`audit`] — Append-only aggregation audit log. **Caller-owned and
//!   experimental**; nothing in the aggregation path writes to it.
//!
//! See `docs/VERIFICATION_INTEGRATION.md` for the policy governing the
//! unintegrated pieces.

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
