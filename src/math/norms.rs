//! Norm computations for vectors.
//!
//! Provides L2 (Euclidean) norm functions used by aggregation and
//! verification modules.

/// Compute the L2 (Euclidean) norm of an f32 slice.
pub fn l2_norm(v: &[f32]) -> f32 {
    l2_norm_sq(v).sqrt()
}

/// Compute the squared L2 norm of an f32 slice (avoids sqrt).
pub fn l2_norm_sq(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_norm_3_4_5() {
        let v = vec![3.0f32, 4.0];
        assert!((l2_norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm_sq_3_4() {
        let v = vec![3.0f32, 4.0];
        assert!((l2_norm_sq(&v) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm_empty() {
        assert_eq!(l2_norm(&[]), 0.0);
        assert_eq!(l2_norm_sq(&[]), 0.0);
    }

    #[test]
    fn test_l2_norm_single() {
        assert!((l2_norm(&[-7.0f32]) - 7.0).abs() < 1e-6);
    }
}
