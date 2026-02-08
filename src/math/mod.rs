//! Mathematical primitives for Qora-FL.
//!
//! Provides shared math utilities used by aggregation and verification:
//!
//! - [`norms`] — L2 norm computations

pub mod norms;

pub use norms::{l2_norm, l2_norm_sq};
