//! Krum safety condition checks.
//!
//! Provides utilities to verify whether the Krum algorithm's
//! theoretical guarantees hold for a given `(n, f)` configuration.

/// Minimum number of clients Krum requires to tolerate `f` Byzantine nodes.
///
/// Returns `2f + 3`, saturating rather than overflowing for absurd `f`
/// (which can reach here from untrusted input such as a parsed `"krum:N"`
/// method string).
pub fn krum_min_clients(f: usize) -> usize {
    f.saturating_mul(2).saturating_add(3)
}

/// Check whether Krum's theoretical guarantee holds: `n >= 2f + 3`.
pub fn krum_condition_met(n: usize, f: usize) -> bool {
    n >= krum_min_clients(f)
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
    fn test_krum_min_clients() {
        assert_eq!(krum_min_clients(0), 3);
        assert_eq!(krum_min_clients(1), 5);
        assert_eq!(krum_min_clients(2), 7);
        // Must saturate rather than overflow on absurd f from untrusted input.
        assert_eq!(krum_min_clients(usize::MAX), usize::MAX);
        assert!(!krum_condition_met(10, usize::MAX));
    }

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
