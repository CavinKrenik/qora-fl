//! Verification primitives for federated learning updates.
//!
//! Provides verification primitives for model updates:
//!
//! - [`krum_condition`] — Krum's `n >= 2f+3` requirement and the Multi-Krum
//!   selection bound. **Wired into aggregation** and enforced.
//! - [`norm_bound`] — L2 norm checking. [`norm_bound::check_norm_bound`] is the
//!   checked API: it validates finiteness and the bound itself, computes the
//!   norm in `f64`, and accepts a norm exactly equal to the bound.
//!
//!   The same comparison backs
//!   [`crate::ByzantineAggregator::with_norm_bound_filter`], which is the
//!   **opt-in** filtering integration -- one implementation, so the standalone
//!   predicate and the aggregation path cannot disagree at the boundary.
//!   Aggregation with no bound configured computes no norms at all.
//!
//!   [`norm_bound::filter_by_norm_bound`] is deprecated and is **not** used by
//!   the aggregation path: it discards verification errors silently and cannot
//!   report per-update reasons.
//! - [`audit`] — Legacy append-only audit log, **deprecated**. Superseded by
//!   [`crate::audit`], which [`crate::ByzantineAggregator::aggregate_with_audit`]
//!   produces. The two serialized shapes are not compatible.
//!
//! See `docs/VERIFICATION_INTEGRATION.md` for the policy behind these
//! decisions.

pub mod audit;
pub mod krum_condition;
pub mod norm_bound;

// Scoped to this one statement so no other deprecated use can hide behind it.
// Callers reaching these through `verification::` still get the warning.
#[allow(deprecated)]
pub use audit::{AggregationAuditEntry, AuditLog};
pub use krum_condition::{krum_condition_met, krum_min_clients, max_multi_krum_m, max_tolerable_f};
pub use norm_bound::check_norm_bound;

// The re-export itself trips the lint; the attribute is scoped to this single
// item so that no other deprecated use can hide behind it. Callers reaching
// `filter_by_norm_bound` through this path still get the deprecation warning.
#[allow(deprecated)]
pub use norm_bound::filter_by_norm_bound;
