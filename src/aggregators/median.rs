//! Coordinate-wise median aggregation
//!
//! Requires strictly fewer than 50% adversarial values per coordinate for an
//! honest majority. Exactly 50% is not strictly fewer than half.

use ndarray::Array2;
use rayon::prelude::*;

use crate::error::QoraError;
use crate::validation::validate_updates;

/// Coordinate-wise median aggregation.
///
/// For each parameter coordinate, computes the median across all client updates.
/// The median is inherently robust to outliers since it selects the middle value.
///
/// # Assumptions
///
/// Requires **strictly fewer than 50%** adversarial values in each coordinate:
/// the median is unchanged only while an honest majority holds. Exactly 50% is
/// not strictly fewer than half and carries no guarantee.
///
/// # Arguments
///
/// * `updates` - Client model updates as 2D arrays (one per client)
///
/// # Errors
///
/// Returns [`QoraError::NonFiniteValue`] if any update contains NaN or
/// infinity. NaN compares unordered, which would silently corrupt the
/// coordinate sort that the median depends on.
pub fn median(updates: &[Array2<f32>]) -> Result<Array2<f32>, QoraError> {
    validate_updates(updates)?;

    let dim = updates[0].dim();
    let (nrows, ncols) = dim;
    let n_params = nrows * ncols;

    let result_vec: Vec<f32> = (0..n_params)
        .into_par_iter()
        .map(|param_idx| {
            let row = param_idx / ncols;
            let col = param_idx % ncols;

            let mut values: Vec<f32> = updates.iter().map(|update| update[[row, col]]).collect();

            // validate_updates guarantees all values are finite, so partial_cmp
            // never returns None here; the fallback is defensive only.
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mid = values.len() / 2;
            if values.len().is_multiple_of(2) {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            }
        })
        .collect();

    Array2::from_shape_vec(dim, result_vec).map_err(|e| QoraError::ShapeError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_median_odd_count() {
        let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];
        let result = median(&updates).unwrap();
        assert_eq!(result[[0, 0]], 2.0);
    }

    #[test]
    fn test_median_even_count() {
        let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]], array![[4.0]]];
        let result = median(&updates).unwrap();
        assert_eq!(result[[0, 0]], 2.5);
    }

    #[test]
    fn test_median_rejects_outlier() {
        let updates = vec![
            array![[1.0]],
            array![[2.0]],
            array![[100.0]], // Outlier
        ];
        let result = median(&updates).unwrap();
        assert_eq!(result[[0, 0]], 2.0);
    }

    #[test]
    fn test_median_30_percent_byzantine() {
        // 7 honest (~1.0), 3 Byzantine (100.0)
        let mut updates: Vec<Array2<f32>> =
            (0..7).map(|i| array![[1.0 + i as f32 * 0.1]]).collect();
        updates.extend(vec![array![[100.0]]; 3]);

        let result = median(&updates).unwrap();
        // Median of sorted [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 100, 100, 100]
        // mid=5, values[4]=1.4, values[5]=1.5 -> median = 1.45
        assert!(
            result[[0, 0]] < 2.0,
            "Expected <2.0, got {}",
            result[[0, 0]]
        );
    }

    #[test]
    fn test_median_empty() {
        let updates: Vec<Array2<f32>> = vec![];
        assert!(median(&updates).is_err());
    }

    #[test]
    fn test_median_dimension_mismatch() {
        let updates = vec![array![[1.0, 2.0]], array![[1.0]]];
        assert!(median(&updates).is_err());
    }

    #[test]
    fn test_median_multi_dimensional() {
        let updates = vec![
            array![[1.0, 10.0], [100.0, 1000.0]],
            array![[2.0, 20.0], [200.0, 2000.0]],
            array![[3.0, 30.0], [300.0, 3000.0]],
        ];
        let result = median(&updates).unwrap();
        assert_eq!(result[[0, 0]], 2.0);
        assert_eq!(result[[0, 1]], 20.0);
        assert_eq!(result[[1, 0]], 200.0);
        assert_eq!(result[[1, 1]], 2000.0);
    }
}
