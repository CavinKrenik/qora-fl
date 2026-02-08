//! Krum safety condition checks.
//!
//! Provides utilities to verify whether the Krum algorithm's
//! theoretical guarantees hold for a given `(n, f)` configuration.

/// Check whether Krum's theoretical guarantee holds: `n >= 2f + 3`.
pub fn krum_condition_met(n: usize, f: usize) -> bool {
    n >= 2 * f + 3
}

/// Compute the maximum number of Byzantine nodes tolerable for `n` clients.
///
/// Returns `(n - 3) / 2` (integer division). Returns 0 if `n < 3`.
pub fn max_tolerable_f(n: usize) -> usize {
    if n < 3 {
        0
    } else {
        (n - 3) / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_krum_condition_met() {
        assert!(krum_condition_met(5, 1)); // 5 >= 2*1+3 = 5
        assert!(krum_condition_met(7, 2)); // 7 >= 2*2+3 = 7
        assert!(krum_condition_met(10, 1)); // 10 >= 5
        assert!(!krum_condition_met(4, 1)); // 4 < 5
        assert!(!krum_condition_met(6, 2)); // 6 < 7
    }

    #[test]
    fn test_max_tolerable_f() {
        assert_eq!(max_tolerable_f(0), 0);
        assert_eq!(max_tolerable_f(2), 0);
        assert_eq!(max_tolerable_f(3), 0); // (3-3)/2 = 0
        assert_eq!(max_tolerable_f(5), 1); // (5-3)/2 = 1
        assert_eq!(max_tolerable_f(7), 2);
        assert_eq!(max_tolerable_f(10), 3);
        assert_eq!(max_tolerable_f(100), 48);
    }
}
