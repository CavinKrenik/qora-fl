//! Coordinate-wise trimmed mean aggregation
//!
//! Achieves Byzantine tolerance by trimming outliers per parameter coordinate.
//! Validated to handle 30% Byzantine attackers in QRES experiments.

use ndarray::Array2;
use rayon::prelude::*;

use crate::error::QoraError;

/// Coordinate-wise trimmed mean aggregation.
///
/// For each parameter coordinate, sorts values across all client updates,
/// trims the top and bottom `trim_fraction` values, then averages the rest.
///
/// # Byzantine Tolerance
///
/// Can handle up to 30% Byzantine clients (validated in QRES).
/// Each coordinate independently trims outliers, so attackers cannot
/// corrupt the aggregate by poisoning individual parameters.
///
/// # Arguments
///
/// * `updates` - Client model updates as 2D arrays (one per client)
/// * `trim_fraction` - Fraction to trim from each end (0.0..0.5, typically 0.2)
///
/// # Note
///
/// Weights are intentionally not supported here because sorting destroys the
/// correspondence between values and their original client weights. Use
/// [`super::fedavg`] for weighted aggregation without Byzantine tolerance.
pub fn trimmed_mean(updates: &[Array2<f32>], trim_fraction: f32) -> Result<Array2<f32>, QoraError> {
    if updates.is_empty() {
        return Err(QoraError::EmptyUpdates);
    }

    if !(0.0..=0.5).contains(&trim_fraction) {
        return Err(QoraError::InvalidTrimFraction(trim_fraction));
    }

    let n_clients = updates.len();
    let dim = updates[0].dim();

    for update in &updates[1..] {
        if update.dim() != dim {
            return Err(QoraError::DimensionMismatch);
        }
    }

    let n_trim = (n_clients as f32 * trim_fraction).ceil() as usize;
    let n_keep = n_clients.saturating_sub(2 * n_trim);

    if n_keep < 1 {
        return Err(QoraError::InsufficientQuorum {
            needed: 1,
            actual: n_keep,
        });
    }

    let (nrows, ncols) = dim;
    let n_params = nrows * ncols;

    let result_vec: Vec<f32> = (0..n_params)
        .into_par_iter()
        .map(|param_idx| {
            let row = param_idx / ncols;
            let col = param_idx % ncols;

            let mut values: Vec<f32> = updates.iter().map(|update| update[[row, col]]).collect();

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let trimmed = &values[n_trim..n_clients - n_trim];
            trimmed.iter().sum::<f32>() / trimmed.len() as f32
        })
        .collect();

    Array2::from_shape_vec(dim, result_vec).map_err(|e| QoraError::ShapeError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_honest_clients_only() {
        let updates = vec![array![[1.0, 2.0]], array![[1.0, 2.0]], array![[1.0, 2.0]]];
        let result = trimmed_mean(&updates, 0.2).unwrap();
        assert!((result[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((result[[0, 1]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_30_percent_byzantine() {
        // 7 honest, 3 Byzantine (30%)
        let mut updates = vec![array![[1.0]]; 7];
        updates.extend(vec![array![[100.0]]; 3]);

        let result = trimmed_mean(&updates, 0.3).unwrap();

        // Should be close to honest mean (1.0), not Byzantine (100.0)
        assert!(
            (result[[0, 0]] - 1.0).abs() < 0.5,
            "Expected ~1.0, got {}",
            result[[0, 0]]
        );
    }

    #[test]
    fn test_empty_updates() {
        let updates: Vec<Array2<f32>> = vec![];
        assert!(trimmed_mean(&updates, 0.2).is_err());
    }

    #[test]
    fn test_invalid_trim_fraction() {
        let updates = vec![array![[1.0]]];
        assert!(trimmed_mean(&updates, 0.6).is_err());
        assert!(trimmed_mean(&updates, -0.1).is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let updates = vec![array![[1.0, 2.0]], array![[1.0]]];
        assert!(trimmed_mean(&updates, 0.2).is_err());
    }

    #[test]
    fn test_multi_dimensional() {
        let updates = vec![
            array![[1.0, 2.0], [3.0, 4.0]],
            array![[1.1, 2.1], [3.1, 4.1]],
            array![[0.9, 1.9], [2.9, 3.9]],
            array![[1.0, 2.0], [3.0, 4.0]],
            array![[50.0, 50.0], [50.0, 50.0]], // Byzantine
        ];
        let result = trimmed_mean(&updates, 0.2).unwrap();
        // Each coordinate should be close to honest mean (~1.0, ~2.0, ~3.0, ~4.0)
        assert!((result[[0, 0]] - 1.0).abs() < 0.15);
        assert!((result[[0, 1]] - 2.0).abs() < 0.15);
        assert!((result[[1, 0]] - 3.0).abs() < 0.15);
        assert!((result[[1, 1]] - 4.0).abs() < 0.15);
    }

    #[test]
    fn test_zero_trim() {
        // trim_fraction=0.0 should behave like mean
        let updates = vec![array![[1.0]], array![[3.0]], array![[5.0]]];
        let result = trimmed_mean(&updates, 0.0).unwrap();
        assert!((result[[0, 0]] - 3.0).abs() < 1e-6);
    }
}
