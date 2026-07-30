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

/// L2 distance between two `f32` sequences, computed entirely in `f64`.
///
/// Each pair is widened to `f64` **before** subtracting. Computing the
/// differences in `f32` first and widening afterwards would be unsound: the
/// subtraction itself can overflow to infinity for large finite operands, and
/// the square can underflow to zero for close ones, so the damage would
/// already be done before any `f64` arithmetic saw the values. For the same
/// reason this cannot be expressed as [`l2_norm_f64`] over a precomputed
/// difference array.
///
/// Iterates the shorter sequence if lengths differ; callers are expected to
/// have validated shapes already.
pub(crate) fn l2_distance_f64<'a, A, B>(a: A, b: B) -> f64
where
    A: IntoIterator<Item = &'a f32>,
    B: IntoIterator<Item = &'a f32>,
{
    a.into_iter()
        .zip(b)
        .map(|(&x, &y)| {
            let difference = f64::from(x) - f64::from(y);
            difference * difference
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

    // --- l2_distance_f64 ---

    #[test]
    fn test_distance_matches_hand_computation() {
        let a = [3.0f32, 0.0];
        let b = [0.0f32, 4.0];
        assert!((l2_distance_f64(a.iter(), b.iter()) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_identical_vectors_give_exactly_zero() {
        let v = [1.5f32, -2.5, 1e20, 1e-25];
        assert_eq!(l2_distance_f64(v.iter(), v.iter()), 0.0);
    }

    /// Large finite operands must not overflow.
    ///
    /// The difference here is 2e20 and its square 4e40, well past `f32::MAX`
    /// (~3.4e38). Subtracting in `f32` would produce `inf` before any `f64`
    /// arithmetic could see the values, which is why the operands are widened
    /// first rather than the result being cast afterwards.
    #[test]
    fn test_large_finite_operands_do_not_overflow() {
        let a = [1e20f32, 1e20];
        let b = [-1e20f32, -1e20];

        let distance = l2_distance_f64(a.iter(), b.iter());

        assert!(distance.is_finite(), "distance overflowed: {}", distance);
        // 2e20 * sqrt(2) ~= 2.828e20
        assert!(
            (distance / 2.828427e20 - 1.0).abs() < 1e-5,
            "got {}",
            distance
        );

        // The f32 route this replaced, shown failing for the record.
        let f32_route: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        assert!(
            f32_route.is_infinite(),
            "f32 accumulation should overflow here; if it no longer does, this \
             test no longer pins anything"
        );
    }

    /// Tiny finite differences must not underflow to a zero distance.
    ///
    /// `(1e-25)^2` is 1e-50, below the smallest `f32` subnormal (~1.4e-45), so
    /// squaring in `f32` flushes every term to zero and reports two distinct
    /// vectors as identical.
    #[test]
    fn test_tiny_finite_differences_do_not_underflow() {
        let a = [1e-25f32, 1e-25];
        let b = [0.0f32, 0.0];

        let distance = l2_distance_f64(a.iter(), b.iter());

        assert!(distance > 0.0, "distance underflowed to {}", distance);
        assert!(
            (distance / 1.414214e-25 - 1.0).abs() < 1e-5,
            "got {}",
            distance
        );

        let f32_route: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        assert_eq!(
            f32_route, 0.0,
            "f32 accumulation should underflow here; if it no longer does, \
             this test no longer pins anything"
        );
    }

    #[test]
    fn test_distance_is_symmetric() {
        let a = [1.0f32, -3.0, 1e18];
        let b = [2.0f32, 5.0, -1e18];
        assert_eq!(
            l2_distance_f64(a.iter(), b.iter()),
            l2_distance_f64(b.iter(), a.iter())
        );
    }
}
