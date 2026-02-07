//! Mathematical primitives for Qora-FL.
//!
//! Fixed-point arithmetic is currently implemented directly in
//! [`crate::aggregators::krum`] using the `I16F16` type from the `fixed` crate.
//! Block floating-point (BFP-16) encoding is also in the Krum module.
//!
//! This module is reserved for shared math utilities if the need arises
//! (e.g., a generic fixed-point conversion layer, norm computations).
