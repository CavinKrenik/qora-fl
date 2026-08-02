//! Norm-bound verification for model updates.
//!
//! Rejects updates whose L2 norm exceeds a threshold, preventing
//! Byzantine clients from injecting extreme gradient values.

use ndarray::Array2;

use crate::error::QoraError;
use crate::math::norms::l2_norm_f64;
use crate::validation::validate_updates;

/// Check that a norm bound is a usable threshold.
///
/// The single definition of the policy: a bound must be finite and strictly
/// positive. Zero and negative bounds reject every possible update, an infinite
/// bound accepts every update regardless of its norm, and a NaN bound rejects
/// every update because every comparison against NaN is false.
///
/// Shared by [`check_norm_bound`], [`evaluate_norm_bound`],
/// [`crate::ByzantineAggregator::with_norm_bound_filter`], and the aggregator's
/// deserialization, so a bound cannot be accepted through one entrance and
/// refused through another.
///
/// # Errors
///
/// [`QoraError::InvalidNormBound`] carrying the offending value.
pub(crate) fn validate_norm_bound(max_norm: f32) -> Result<(), QoraError> {
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Err(QoraError::InvalidNormBound { value: max_norm });
    }
    Ok(())
}

/// Measure an update against a norm bound, returning the norm it computed.
///
/// The checked primitive behind both [`check_norm_bound`] and the aggregator's
/// optional norm filtering. It exists so that there is exactly one
/// implementation of the norm calculation and the comparison, and so that an
/// integration can record *why* an update was excluded without re-deriving the
/// measurement or parsing a rendered error message.
///
/// The norm is computed in `f64` via [`l2_norm_f64`] and compared in `f64`, so
/// the check is exact across the full `f32` range. Computing it in `f32` is
/// not viable: the squared sum overflows to infinity above ~3.4e38 and
/// underflows to zero below ~1.4e-45, which respectively rejects valid updates
/// and accepts oversized ones as having no norm at all.
///
/// Equality is acceptance: a norm exactly equal to the bound passes. The bound
/// is the largest permitted norm, not the first forbidden one.
///
/// # Order of checks
///
/// The update is validated before the bound. A call carrying both a poisoned
/// update and an unusable bound reports the poisoned update, because that is
/// the finding a caller must act on first -- an invalid bound is the caller's
/// own misconfiguration, while a non-finite coordinate is evidence about a
/// client. This ordering is pinned by test.
///
/// # Errors
///
/// * [`QoraError::NonFiniteValue`] if the update contains NaN or infinity.
/// * [`QoraError::InvalidNormBound`] if `max_norm` is not finite and > 0.
/// * [`QoraError::NormBoundExceeded`] if a valid update exceeds a valid bound,
///   carrying both the computed norm and the bound.
pub(crate) fn evaluate_norm_bound(update: &Array2<f32>, max_norm: f32) -> Result<f64, QoraError> {
    validate_updates(std::slice::from_ref(update))?;
    validate_norm_bound(max_norm)?;

    let flat: Vec<f32> = update.iter().copied().collect();
    let norm = l2_norm_f64(&flat);

    if norm <= max_norm as f64 {
        Ok(norm)
    } else {
        Err(QoraError::NormBoundExceeded {
            norm,
            bound: max_norm,
        })
    }
}

/// Check that a model update's L2 norm is within `max_norm`.
///
/// Delegates to [`evaluate_norm_bound`] and discards the accepted norm, so the
/// predicate and the filtering path can never disagree -- including at the
/// boundary, where a norm exactly equal to `max_norm` is accepted by both.
///
/// # Order of checks
///
/// The update is validated before the bound; see [`evaluate_norm_bound`].
///
/// # Errors
///
/// * [`QoraError::NonFiniteValue`] if the update contains NaN or infinity.
///   `update_index` is always 0: this function inspects a single update and has
///   no slice to index into. `row` and `col` locate the offending coordinate,
///   so a caller iterating a batch can attribute it.
/// * [`QoraError::InvalidNormBound`] if `max_norm` is not finite and > 0.
/// * [`QoraError::NormBoundExceeded`] if a valid update exceeds a valid bound.
///   This was a free-form [`QoraError::VerificationError`] before 0.4.0; the
///   norm and the bound are now structural fields.
pub fn check_norm_bound(update: &Array2<f32>, max_norm: f32) -> Result<(), QoraError> {
    evaluate_norm_bound(update, max_norm).map(|_| ())
}

/// Filter updates by norm bound, returning indices of those that pass.
#[deprecated(
    note = "silently discards verification errors; call check_norm_bound for each update and handle errors explicitly"
)]
pub fn filter_by_norm_bound(updates: &[Array2<f32>], max_norm: f32) -> Vec<usize> {
    updates
        .iter()
        .enumerate()
        .filter(|(_, u)| check_norm_bound(u, max_norm).is_ok())
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_norm_bound_pass() {
        let update = array![[1.0, 0.0]]; // norm = 1.0
        assert!(check_norm_bound(&update, 2.0).is_ok());
    }

    #[test]
    fn test_norm_bound_fail() {
        let update = array![[3.0, 4.0]]; // norm = 5.0
        match check_norm_bound(&update, 4.0) {
            Err(QoraError::NormBoundExceeded { norm, bound }) => {
                assert!((norm - 5.0).abs() < 1e-12, "got {}", norm);
                assert_eq!(bound, 4.0);
            }
            other => panic!("expected NormBoundExceeded, got {:?}", other),
        }
    }

    #[test]
    fn test_norm_bound_exact() {
        let update = array![[3.0, 4.0]]; // norm = 5.0
        assert!(check_norm_bound(&update, 5.0).is_ok());
    }

    /// The bound is the largest permitted norm, not the first forbidden one.
    ///
    /// Pinned across all three positions because the aggregator's optional
    /// filtering shares this comparison; a disagreement at the boundary between
    /// the public checker and the integrated filter would be invisible
    /// everywhere else.
    #[test]
    fn boundary_is_inclusive_below_at_and_above() {
        let update = array![[3.0, 4.0]]; // norm = 5.0 exactly
        assert!(check_norm_bound(&update, 5.5).is_ok(), "below the bound");
        assert!(check_norm_bound(&update, 5.0).is_ok(), "exactly the bound");
        assert!(matches!(
            check_norm_bound(&update, 4.999_999),
            Err(QoraError::NormBoundExceeded { .. })
        ));
    }

    /// The accepted norm is returned, not just a pass/fail verdict.
    ///
    /// This is what lets the aggregator record a measurement without recomputing
    /// it or reading it back out of an error message.
    #[test]
    fn evaluate_returns_the_measured_norm() {
        assert!((evaluate_norm_bound(&array![[3.0, 4.0]], 10.0).unwrap() - 5.0).abs() < 1e-12);

        match evaluate_norm_bound(&array![[3.0, 4.0]], 1.0) {
            Err(QoraError::NormBoundExceeded { norm, bound }) => {
                assert!((norm - 5.0).abs() < 1e-12);
                assert_eq!(bound, 1.0);
            }
            other => panic!("expected NormBoundExceeded, got {:?}", other),
        }
    }

    /// One policy for what counts as a usable bound.
    #[test]
    fn validate_norm_bound_matches_the_checker() {
        let update = array![[1.0, 0.0]];
        for bound in [1e-30, 1.0, 1e30, f32::MIN_POSITIVE, f32::MAX] {
            assert!(validate_norm_bound(bound).is_ok(), "bound {}", bound);
        }
        for bad in [0.0, -1.0, f32::NEG_INFINITY, f32::NAN, f32::INFINITY] {
            assert!(validate_norm_bound(bad).is_err(), "bound {}", bad);
            assert!(
                matches!(
                    check_norm_bound(&update, bad),
                    Err(QoraError::InvalidNormBound { .. })
                ),
                "checker must agree about bound {}",
                bad
            );
        }
    }

    // --- Regression coverage for docs/VERIFICATION_INTEGRATION.md section 6 ---
    //
    // These tests pin the corrected contract of `check_norm_bound`. Several of
    // them fail against the pre-fix implementation; each records why.

    /// Non-finite input is a validation failure, not a bound violation.
    ///
    /// Pre-fix this returned `VerificationError("Update norm NaN exceeds bound
    /// 10.0000")` -- a false statement that also misattributes a malformed
    /// update as a policy failure, leaving a caller matching on `QoraError`
    /// unable to tell the two apart.
    ///
    /// `update_index` is 0 because `check_norm_bound` inspects a single update
    /// and has no slice to index into; callers working over a batch map the
    /// position back themselves.
    #[test]
    fn test_nan_returns_non_finite_value() {
        let update = array![[1.0, f32::NAN]];
        match check_norm_bound(&update, 10.0) {
            Err(QoraError::NonFiniteValue {
                update_index,
                row,
                col,
                value,
            }) => {
                assert_eq!(update_index, 0);
                assert_eq!((row, col), (0, 1));
                assert!(value.is_nan(), "expected NaN, got {}", value);
            }
            other => panic!("expected NonFiniteValue, got {:?}", other),
        }
    }

    /// Both infinities are rejected the same way, and are attributed to the
    /// coordinate that carries them.
    #[test]
    fn test_infinities_return_non_finite_value() {
        for bad in [f32::INFINITY, f32::NEG_INFINITY] {
            let update = array![[0.0, 1.0], [bad, 2.0]];
            match check_norm_bound(&update, 10.0) {
                Err(QoraError::NonFiniteValue {
                    update_index,
                    row,
                    col,
                    value,
                }) => {
                    assert_eq!(update_index, 0);
                    assert_eq!((row, col), (1, 0), "should locate {}", bad);
                    assert_eq!(value, bad);
                }
                other => panic!("expected NonFiniteValue for {}, got {:?}", bad, other),
            }
        }
    }

    /// A finite update well inside its bound must be accepted.
    ///
    /// True norm is `1e20 * sqrt(2)` ~= 1.414e20, four orders of magnitude
    /// below the 1e30 bound. Pre-fix this was **rejected**: `l2_norm_sq`
    /// accumulates in f32, so the squared sum (1e40) exceeds `f32::MAX`
    /// (~3.4e38) and saturates to `inf`, which exceeds every finite bound.
    #[test]
    fn test_large_finite_vector_within_bound_is_accepted() {
        let update = array![[1e20f32, 1e20f32]];
        assert!(
            check_norm_bound(&update, 1e30).is_ok(),
            "1e20 vector (norm ~1.41e20) must pass a 1e30 bound; \
             a failure here means the squared norm overflowed f32"
        );
    }

    /// The same vector must still be rejected by a bound it genuinely exceeds.
    ///
    /// Guards against over-correcting the overflow into blanket acceptance.
    #[test]
    fn test_large_finite_vector_outside_bound_is_rejected() {
        let update = array![[1e20f32, 1e20f32]];
        assert!(matches!(
            check_norm_bound(&update, 1e19),
            Err(QoraError::NormBoundExceeded { .. })
        ));
    }

    /// A rejection must report the update's real norm.
    ///
    /// Pre-fix the message read "Update norm inf exceeds bound ...", because
    /// the overflow reached the formatter. The rejection was correct by
    /// accident; the number justifying it was not.
    #[test]
    fn test_rejection_reports_finite_norm() {
        let update = array![[1e20f32, 1e20f32]];
        match check_norm_bound(&update, 1e19) {
            Err(QoraError::NormBoundExceeded { norm, .. }) => assert!(
                norm.is_finite() && (norm / 1.414_213_5e20 - 1.0).abs() < 1e-6,
                "rejection must carry the real norm (~1.41e20), got: {}",
                norm
            ),
            other => panic!("expected NormBoundExceeded, got {:?}", other),
        }
    }

    /// Bounds that cannot express a threshold are caller misconfiguration.
    ///
    /// Pre-fix all four were silently absorbed: zero and negative bounds
    /// rejected everything as if oversized, NaN rejected everything because
    /// every comparison against NaN is false, and `INFINITY` **accepted
    /// everything** -- a bound that verifies nothing while appearing to.
    #[test]
    fn test_invalid_bounds_are_rejected() {
        let update = array![[1.0, 0.0]]; // norm = 1.0, unremarkable
        for bad in [0.0, -1.0, f32::NEG_INFINITY, f32::NAN, f32::INFINITY] {
            match check_norm_bound(&update, bad) {
                Err(QoraError::InvalidNormBound { value }) => {
                    assert_eq!(value.is_nan(), bad.is_nan());
                    if !bad.is_nan() {
                        assert_eq!(value, bad);
                    }
                }
                other => panic!("bound {} must be InvalidNormBound, got {:?}", bad, other),
            }
        }
    }

    /// Very small finite values must not underflow to a zero norm.
    ///
    /// `(1e-25)^2` is 1e-50, below the smallest f32 subnormal (~1.4e-45), so
    /// pre-fix every term underflowed to 0 and the norm came back as exactly
    /// 0.0. That is a **false accept**: the true norm (~1.41e-25) exceeds the
    /// 1e-26 bound, but a zero norm passes any positive bound. Same root cause
    /// as the overflow, opposite direction, and this one fails open.
    #[test]
    fn test_small_finite_values_do_not_underflow() {
        let update = array![[1e-25f32, 1e-25f32]]; // true norm ~1.414e-25

        assert!(
            check_norm_bound(&update, 1e-26).is_err(),
            "norm ~1.41e-25 exceeds a 1e-26 bound; acceptance means the \
             squared terms underflowed to zero"
        );

        assert!(
            check_norm_bound(&update, 1e-24).is_ok(),
            "the same vector is comfortably inside a 1e-24 bound"
        );
    }

    /// Validation precedes bound checking: a poisoned update is reported even
    /// when the bound is also unusable.
    #[test]
    fn non_finite_update_precedes_invalid_bound_error() {
        let update = array![[1.0, f32::NAN]];
        for bad_bound in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    check_norm_bound(&update, bad_bound),
                    Err(QoraError::NonFiniteValue { .. })
                ),
                "NaN update must outrank bound {}",
                bad_bound
            );
        }
    }

    /// With a well-formed update, an unusable bound is the reported failure --
    /// never a `VerificationError`, which would imply the update was at fault.
    #[test]
    fn finite_update_with_invalid_bound_returns_invalid_norm_bound() {
        let update = array![[3.0, 4.0]]; // norm = 5.0, entirely well-formed
        for bad_bound in [0.0, -1.0, f32::NEG_INFINITY, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    check_norm_bound(&update, bad_bound),
                    Err(QoraError::InvalidNormBound { .. })
                ),
                "bound {} must be InvalidNormBound",
                bad_bound
            );
        }
    }

    /// The norm is computed in f64 and stays exact at both extremes.
    #[test]
    fn test_reported_norm_is_exact_at_both_extremes() {
        // Large: true norm 1e20 * sqrt(2) ~= 1.414214e20.
        match check_norm_bound(&array![[1e20f32, 1e20f32]], 1.0) {
            Err(QoraError::NormBoundExceeded { norm, .. }) => {
                assert!((norm / 1.414_213_5e20 - 1.0).abs() < 1e-6, "got: {}", norm)
            }
            other => panic!("expected NormBoundExceeded, got {:?}", other),
        }

        // Small: true norm 1e-25 * sqrt(2) ~= 1.414214e-25, which must be a
        // positive number rather than the pre-fix 0.0.
        match check_norm_bound(&array![[1e-25f32, 1e-25f32]], 1e-26) {
            Err(QoraError::NormBoundExceeded { norm, .. }) => {
                assert!((norm / 1.414_213_5e-25 - 1.0).abs() < 1e-6, "got: {}", norm)
            }
            other => panic!("expected NormBoundExceeded, got {:?}", other),
        }
    }

    /// The rendered message still names both measurements.
    ///
    /// Scientific notation stays readable across the range this check covers;
    /// `{:.4}` printed a 1e20 norm as 20 leading digits.
    #[test]
    fn rejection_message_names_the_norm_and_the_bound() {
        let err = check_norm_bound(&array![[1e20f32, 1e20f32]], 1e19).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("1.414214e20"), "got: {}", msg);
        assert!(msg.contains("1.000000e19"), "got: {}", msg);
    }

    // Deprecated, but still shipped, so its behavior stays pinned.
    #[allow(deprecated)]
    #[test]
    fn test_filter_by_norm_bound() {
        let updates = vec![
            array![[1.0, 0.0]],   // norm = 1.0
            array![[3.0, 4.0]],   // norm = 5.0
            array![[0.5, 0.5]],   // norm ≈ 0.707
            array![[10.0, 10.0]], // norm ≈ 14.14
        ];
        let passing = filter_by_norm_bound(&updates, 2.0);
        assert_eq!(passing, vec![0, 2]);
    }
}
