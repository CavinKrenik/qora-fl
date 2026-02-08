//! Norm-bound verification for model updates.
//!
//! Rejects updates whose L2 norm exceeds a threshold, preventing
//! Byzantine clients from injecting extreme gradient values.

use ndarray::Array2;

use crate::error::QoraError;
use crate::math::norms::l2_norm;

/// Check that a model update's L2 norm is within `max_norm`.
///
/// Returns `Ok(())` if the norm is within bounds, or a
/// [`QoraError::VerificationError`] if it exceeds the limit.
pub fn check_norm_bound(update: &Array2<f32>, max_norm: f32) -> Result<(), QoraError> {
    let flat: Vec<f32> = update.iter().copied().collect();
    let norm = l2_norm(&flat);
    if norm <= max_norm {
        Ok(())
    } else {
        Err(QoraError::VerificationError(format!(
            "Update norm {:.4} exceeds bound {:.4}",
            norm, max_norm
        )))
    }
}

/// Filter updates by norm bound, returning indices of those that pass.
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
        let result = check_norm_bound(&update, 4.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            QoraError::VerificationError(_)
        ));
    }

    #[test]
    fn test_norm_bound_exact() {
        let update = array![[3.0, 4.0]]; // norm = 5.0
        assert!(check_norm_bound(&update, 5.0).is_ok());
    }

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
