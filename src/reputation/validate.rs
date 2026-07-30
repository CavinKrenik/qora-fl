//! Numeric rules for reputation state.
//!
//! One invariant governs the whole module:
//!
//! > Every stored reputation score is finite and within `[0.0, 1.0]`, and
//! > every arithmetic operation preserves that.
//!
//! These are domain rules rather than general input validation, so they live
//! beside the store they protect rather than in [`crate::validation`].
//!
//! # Why rejection rather than clamping
//!
//! The previous implementation clamped, and the clamps were the vulnerability.
//! `f32::clamp` propagates NaN, so `set_score(id, NaN)` stored NaN. `f32::min`
//! and `f32::max` *discard* NaN, so `reward(id, NaN)` silently produced `1.0`
//! (fully trusted) and `penalize(id, NaN)` produced `0.0` (banned) -- plausible
//! extremes with no NaN left for an operator to notice. Neither clamp
//! constrained the *sign* of an amount, so `reward(id, -5.0)` stored `-4.5`.
//!
//! Rejecting invalid input is the only outcome that neither corrupts the store
//! nor hides the caller's mistake.
//!
//! # Checking `is_finite` explicitly
//!
//! `(0.0..=1.0).contains(&value)` is already false for NaN, so a range check
//! alone would reject it. It is spelled out anyway: the two failures deserve
//! different diagnostics, and relying on NaN's comparison behaviour to enforce
//! a range is exactly the implicit reasoning that produced these bugs.

use crate::error::QoraError;

/// Validate a score for storage: finite and within `[0.0, 1.0]`.
pub(crate) fn validate_score(value: f32) -> Result<(), QoraError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(QoraError::InvalidReputationScore { value });
    }
    Ok(())
}

/// Validate a reward or penalty amount: finite and non-negative.
///
/// Amounts above 1.0 are permitted; the resulting score saturates at the
/// boundary. Only the sign and finiteness are constrained, because the
/// magnitude cannot break the invariant once the result is clamped.
pub(crate) fn validate_adjustment(value: f32) -> Result<(), QoraError> {
    if !value.is_finite() || value < 0.0 {
        return Err(QoraError::InvalidReputationAdjustment { value });
    }
    Ok(())
}

/// Validate a decay factor: finite and within `[0.0, 1.0]`.
pub(crate) fn validate_decay_factor(value: f32) -> Result<(), QoraError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(QoraError::InvalidReputationDecay { value });
    }
    Ok(())
}

/// Validate a ban threshold: finite and within `[0.0, 1.0]`.
pub(crate) fn validate_threshold(value: f32) -> Result<(), QoraError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(QoraError::InvalidReputationThreshold { value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const NON_FINITE: [f32; 3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

    #[test]
    fn scores_accept_the_closed_unit_interval() {
        for good in [0.0, 0.5, 1.0, f32::MIN_POSITIVE] {
            assert!(validate_score(good).is_ok(), "{} should be storable", good);
        }
    }

    #[test]
    fn scores_reject_non_finite_and_out_of_range() {
        for bad in NON_FINITE {
            assert!(matches!(
                validate_score(bad),
                Err(QoraError::InvalidReputationScore { .. })
            ));
        }
        for bad in [-0.1, 1.1, -1.0, 5.0] {
            assert!(
                matches!(
                    validate_score(bad),
                    Err(QoraError::InvalidReputationScore { .. })
                ),
                "{} should be rejected, not clamped",
                bad
            );
        }
    }

    #[test]
    fn adjustments_accept_any_finite_non_negative_magnitude() {
        for good in [0.0, 0.02, 1.0, 5.0, 1e30] {
            assert!(
                validate_adjustment(good).is_ok(),
                "{} should be a usable amount",
                good
            );
        }
    }

    #[test]
    fn adjustments_reject_non_finite_and_negative() {
        for bad in NON_FINITE {
            assert!(matches!(
                validate_adjustment(bad),
                Err(QoraError::InvalidReputationAdjustment { .. })
            ));
        }
        for bad in [-0.001, -5.0] {
            assert!(
                matches!(
                    validate_adjustment(bad),
                    Err(QoraError::InvalidReputationAdjustment { .. })
                ),
                "{} inverts the operation and must be rejected",
                bad
            );
        }
    }

    #[test]
    fn decay_factors_are_bounded_to_the_unit_interval() {
        for good in [0.0, 0.05, 1.0] {
            assert!(validate_decay_factor(good).is_ok());
        }
        for bad in [-0.1, 1.1] {
            assert!(matches!(
                validate_decay_factor(bad),
                Err(QoraError::InvalidReputationDecay { .. })
            ));
        }
        for bad in NON_FINITE {
            assert!(matches!(
                validate_decay_factor(bad),
                Err(QoraError::InvalidReputationDecay { .. })
            ));
        }
    }

    #[test]
    fn thresholds_allow_values_above_the_default_score() {
        // A threshold above 0.5 deliberately rejects unknown clients.
        for good in [0.0, 0.2, 0.5, 0.99, 1.0] {
            assert!(
                validate_threshold(good).is_ok(),
                "{} is a legitimate threshold",
                good
            );
        }
        for bad in [-0.1, 1.1] {
            assert!(matches!(
                validate_threshold(bad),
                Err(QoraError::InvalidReputationThreshold { .. })
            ));
        }
        for bad in NON_FINITE {
            assert!(matches!(
                validate_threshold(bad),
                Err(QoraError::InvalidReputationThreshold { .. })
            ));
        }
    }
}
