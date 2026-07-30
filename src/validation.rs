//! Shared input validation.
//!
//! Cross-cutting infrastructure, private to the crate and deliberately placed
//! at the root rather than under `aggregators`: both `aggregators` and
//! `verification` need it, and `aggregators` already depends on
//! `verification`, so hosting it under either one would create a cycle.
//!
//! Every aggregation entry point runs [`validate_updates`] before touching
//! client data, as does [`crate::verification::check_norm_bound`].
//! Centralizing the checks here means the guarantee holds regardless of
//! whether callers use [`crate::ByzantineAggregator`] or call the free
//! functions ([`crate::trimmed_mean`], [`crate::median`], [`crate::fedavg`])
//! directly.
//!
//! # Why non-finite values must be rejected, not sanitized
//!
//! Silently coercing NaN to a finite value is unsafe for Byzantine-tolerant
//! aggregation. The comparator used by the coordinate-wise methods
//! (`partial_cmp(...).unwrap_or(Equal)`) treats NaN as equal to every other
//! value, so a NaN's position after sorting depends on which client submitted
//! it -- meaning the aggregate silently becomes order-dependent. Worse, the
//! BFP-16 encoder quantizes NaN to a zero mantissa, so Krum's distance metric
//! reads a poisoned coordinate as `0.0`, which can make an attacker look
//! *closer* to the honest cluster than a genuine outlier. Rejecting the round
//! is the only outcome that does not silently corrupt the result.
//!
//! Non-finite values are also invisible to reputation tracking: a NaN distance
//! satisfies neither `distance < 1.0` nor `distance > 10.0`, so a NaN attacker
//! is never penalized and never banned. Validation must therefore run *before*
//! reputation gating, not alongside it.

use ndarray::Array2;

use crate::error::QoraError;

/// Validate a batch of client updates before aggregation.
///
/// Checks, in order:
/// 1. The batch is non-empty ([`QoraError::EmptyUpdates`]).
/// 2. Every update has the same shape ([`QoraError::DimensionMismatch`]).
/// 3. Every value is finite ([`QoraError::NonFiniteValue`]).
///
/// The non-finite error reports the positional index of the offending update
/// and the `[row, col]` of the offending value.
pub fn validate_updates(updates: &[Array2<f32>]) -> Result<(), QoraError> {
    if updates.is_empty() {
        return Err(QoraError::EmptyUpdates);
    }

    let dim = updates[0].dim();

    for (update_index, update) in updates.iter().enumerate() {
        if update.dim() != dim {
            return Err(QoraError::DimensionMismatch);
        }

        for ((row, col), &value) in update.indexed_iter() {
            if !value.is_finite() {
                return Err(QoraError::NonFiniteValue {
                    update_index,
                    row,
                    col,
                    value,
                });
            }
        }
    }

    Ok(())
}

/// Validate that every client weight is finite.
///
/// A single NaN or infinite weight corrupts a weighted mean even when all
/// updates are well-formed, so weights get the same treatment as updates.
pub fn validate_weights(weights: &[f32]) -> Result<(), QoraError> {
    for (index, &value) in weights.iter().enumerate() {
        if !value.is_finite() {
            return Err(QoraError::NonFiniteWeight { index, value });
        }
    }
    Ok(())
}

/// Validate that `client_ids` (when supplied) lines up 1:1 with `updates`.
///
/// A length mismatch would otherwise index out of bounds during reputation
/// gating, turning caller error into a panic.
pub fn validate_client_ids<T>(
    updates: &[Array2<f32>],
    client_ids: Option<&[T]>,
) -> Result<(), QoraError> {
    if let Some(ids) = client_ids {
        if ids.len() != updates.len() {
            return Err(QoraError::ClientIdCountMismatch {
                updates: updates.len(),
                client_ids: ids.len(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_accepts_valid_updates() {
        let updates = vec![array![[1.0, 2.0]], array![[3.0, 4.0]]];
        assert!(validate_updates(&updates).is_ok());
    }

    #[test]
    fn test_rejects_empty() {
        let updates: Vec<Array2<f32>> = vec![];
        assert!(matches!(
            validate_updates(&updates),
            Err(QoraError::EmptyUpdates)
        ));
    }

    #[test]
    fn test_rejects_dimension_mismatch() {
        let updates = vec![array![[1.0, 2.0]], array![[1.0]]];
        assert!(matches!(
            validate_updates(&updates),
            Err(QoraError::DimensionMismatch)
        ));
    }

    #[test]
    fn test_rejects_nan_and_reports_location() {
        let updates = vec![array![[1.0, 2.0]], array![[3.0, f32::NAN]]];
        match validate_updates(&updates) {
            Err(QoraError::NonFiniteValue {
                update_index,
                row,
                col,
                value,
            }) => {
                assert_eq!(update_index, 1);
                assert_eq!((row, col), (0, 1));
                assert!(value.is_nan());
            }
            other => panic!("expected NonFiniteValue, got {:?}", other),
        }
    }

    #[test]
    fn test_rejects_positive_and_negative_infinity() {
        for bad in [f32::INFINITY, f32::NEG_INFINITY] {
            let updates = vec![array![[1.0]], array![[bad]]];
            assert!(
                matches!(
                    validate_updates(&updates),
                    Err(QoraError::NonFiniteValue {
                        update_index: 1,
                        ..
                    })
                ),
                "should reject {}",
                bad
            );
        }
    }

    #[test]
    fn test_detects_non_finite_in_any_update_position() {
        // NaN must be rejected regardless of which update carries it -- this is
        // what made trimmed_mean's old behavior order-dependent.
        for nan_pos in 0..5 {
            let mut updates: Vec<Array2<f32>> = (0..5).map(|_| array![[1.0f32]]).collect();
            updates[nan_pos] = array![[f32::NAN]];
            assert!(
                matches!(
                    validate_updates(&updates),
                    Err(QoraError::NonFiniteValue { update_index, .. }) if update_index == nan_pos
                ),
                "NaN in update {} should be rejected and attributed",
                nan_pos
            );
        }
    }

    #[test]
    fn test_multi_dimensional_row_col() {
        let updates = vec![array![[1.0, 2.0], [3.0, f32::NAN]]];
        match validate_updates(&updates) {
            Err(QoraError::NonFiniteValue { row, col, .. }) => assert_eq!((row, col), (1, 1)),
            other => panic!("expected NonFiniteValue, got {:?}", other),
        }
    }

    #[test]
    fn test_weights_accepts_finite() {
        assert!(validate_weights(&[1.0, 0.0, -2.5]).is_ok());
        assert!(validate_weights(&[]).is_ok());
    }

    #[test]
    fn test_weights_rejects_non_finite() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                matches!(
                    validate_weights(&[1.0, bad, 3.0]),
                    Err(QoraError::NonFiniteWeight { index: 1, .. })
                ),
                "should reject weight {}",
                bad
            );
        }
    }

    #[test]
    fn test_client_ids_none_is_ok() {
        let updates = vec![array![[1.0]]];
        assert!(validate_client_ids::<String>(&updates, None).is_ok());
    }

    #[test]
    fn test_client_ids_matching_length_is_ok() {
        let updates = vec![array![[1.0]], array![[2.0]]];
        let ids = vec!["a".to_string(), "b".to_string()];
        assert!(validate_client_ids(&updates, Some(&ids)).is_ok());
    }

    #[test]
    fn test_client_ids_too_many_and_too_few() {
        let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];

        let too_many: Vec<String> = (0..6).map(|i| format!("c{}", i)).collect();
        assert!(matches!(
            validate_client_ids(&updates, Some(&too_many)),
            Err(QoraError::ClientIdCountMismatch {
                updates: 3,
                client_ids: 6
            })
        ));

        let too_few = vec!["c0".to_string()];
        assert!(matches!(
            validate_client_ids(&updates, Some(&too_few)),
            Err(QoraError::ClientIdCountMismatch {
                updates: 3,
                client_ids: 1
            })
        ));
    }
}
