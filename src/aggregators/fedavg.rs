//! FedAvg baseline aggregation (no Byzantine tolerance)
//!
//! Standard federated averaging as described by McMahan et al. (2017).
//! Vulnerable to even a single Byzantine client. Included as a baseline
//! for comparison against Byzantine-tolerant methods.

use ndarray::Array2;

use crate::error::QoraError;
use crate::validation::{validate_updates, validate_weights};

/// Standard FedAvg aggregation (no Byzantine defense).
///
/// Computes a (optionally weighted) arithmetic mean across client updates.
/// A single malicious client can corrupt the aggregate.
///
/// # Arguments
///
/// * `updates` - Client model updates as 2D arrays (one per client)
/// * `weights` - Optional client weights (e.g., proportional to dataset size)
///
/// # Errors
///
/// Returns [`QoraError::NonFiniteValue`] if any update contains NaN or
/// infinity, or [`QoraError::NonFiniteWeight`] if any weight does. FedAvg
/// propagates non-finite values directly into every output coordinate, so a
/// single bad value from one client -- or a single bad weight -- destroys the
/// entire aggregate.
pub fn fedavg(updates: &[Array2<f32>], weights: Option<&[f32]>) -> Result<Array2<f32>, QoraError> {
    validate_updates(updates)?;

    let dim = updates[0].dim();

    match weights {
        Some(w) => {
            if w.len() != updates.len() {
                return Err(QoraError::DimensionMismatch);
            }
            validate_weights(w)?;
            let weight_sum: f32 = w.iter().sum();
            if weight_sum == 0.0 {
                return Err(QoraError::InsufficientQuorum {
                    needed: 1,
                    actual: 0,
                });
            }
            let weighted_sum = updates
                .iter()
                .zip(w.iter())
                .fold(Array2::<f32>::zeros(dim), |acc, (update, &weight)| {
                    acc + &(update * weight)
                });
            Ok(weighted_sum / weight_sum)
        }
        None => {
            let n = updates.len() as f32;
            let sum = updates
                .iter()
                .fold(Array2::<f32>::zeros(dim), |acc, update| acc + update);
            Ok(sum / n)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fedavg_unweighted() {
        let updates = vec![array![[1.0, 2.0]], array![[3.0, 4.0]]];
        let result = fedavg(&updates, None).unwrap();
        assert!((result[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((result[[0, 1]] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_fedavg_weighted() {
        let updates = vec![array![[1.0]], array![[3.0]]];
        let weights = vec![1.0, 3.0]; // Second client has 3x weight
        let result = fedavg(&updates, Some(&weights)).unwrap();
        // (1*1 + 3*3) / 4 = 10/4 = 2.5
        assert!((result[[0, 0]] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_fedavg_vulnerable_to_attack() {
        let updates = vec![
            array![[1.0]],
            array![[1.0]],
            array![[100.0]], // Single attacker poisons result
        ];
        let result = fedavg(&updates, None).unwrap();
        // FedAvg is corrupted: (1 + 1 + 100) / 3 = 34.0
        assert!(
            result[[0, 0]] > 10.0,
            "FedAvg should be corrupted by attacker"
        );
    }

    #[test]
    fn test_fedavg_empty() {
        let updates: Vec<Array2<f32>> = vec![];
        assert!(fedavg(&updates, None).is_err());
    }

    #[test]
    fn test_fedavg_weight_length_mismatch() {
        let updates = vec![array![[1.0]], array![[2.0]]];
        let weights = vec![1.0]; // Wrong length
        assert!(fedavg(&updates, Some(&weights)).is_err());
    }

    #[test]
    fn test_fedavg_single_client() {
        let updates = vec![array![[42.0, 7.0]]];
        let result = fedavg(&updates, None).unwrap();
        assert_eq!(result[[0, 0]], 42.0);
        assert_eq!(result[[0, 1]], 7.0);
    }
}
