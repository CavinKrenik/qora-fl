//! Norm computations for vectors.
//!
//! Provides L2 (Euclidean) norm functions used by aggregation and
//! verification modules.

/// Compute the L2 (Euclidean) norm of an f32 slice.
///
/// Accumulates in `f64` internally, so the squared sum does not overflow or
/// underflow for any input whose true norm is representable as an `f32`. The
/// returned value still saturates to infinity if the true norm itself exceeds
/// `f32::MAX`.
pub fn l2_norm(v: &[f32]) -> f32 {
    l2_norm_f64(v) as f32
}

/// Compute the L2 (Euclidean) norm of an f32 slice, accumulating in `f64`.
///
/// Squaring an `f32` in `f32` is unsafe at the extremes: the squared sum
/// overflows to infinity above ~3.4e38 and flushes to zero below ~1.4e-45,
/// even when the true norm sits comfortably inside `f32` range. Both are
/// silent, and for norm-bound verification both are exploitable -- an
/// underflowed norm of `0.0` passes every bound.
///
/// `f64` carries the squared sum with room to spare: the largest possible
/// term is `f32::MAX^2` ~= 1.2e77, far below `f64::MAX` (~1.8e308), so a
/// `Vec<f32>` large enough to overflow the sum cannot fit in memory.
pub(crate) fn l2_norm_f64(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| {
            let x = x as f64;
            x * x
        })
        .sum::<f64>()
        .sqrt()
}

/// Compute the squared L2 norm of an f32 slice (avoids sqrt).
///
/// # Range limitation
///
/// The `f32` return type cannot represent a squared norm outside `f32` range,
/// so the result **underflows to zero** when the true squared norm is below
/// ~1.4e-45 and **overflows to infinity** above ~3.4e38 -- both silently. A
/// vector of `1e-25` values squares to `1e-50` and reports `0.0`; a vector of
/// `1e20` values squares to `1e40` and reports `inf`.
///
/// This is a property of the return type, not of the accumulation, and cannot
/// be fixed without changing the signature. Do not use this function for
/// verification or any other decision that must hold at extreme scales; use
/// [`l2_norm`], which accumulates in `f64`.
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
