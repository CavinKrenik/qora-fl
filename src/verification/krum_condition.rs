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

/// Maximum number of vectors Multi-Krum may select for a given `(n, f)`.
///
/// Returns `n - 2f - 2` (Blanchard et al., 2017), or `0` when the Krum quorum
/// condition `n >= 2f + 3` is not met -- selecting any number of vectors is
/// unsound in that case, so there is no safe maximum to report.
///
/// Whenever the quorum condition *is* met the result is at least 1, since
/// `n >= 2f + 3` implies `n - 2f - 2 >= 1`. A caller that has already checked
/// [`krum_condition_met`] can therefore rely on a usable value.
pub fn max_multi_krum_m(n: usize, f: usize) -> usize {
    if !krum_condition_met(n, f) {
        return 0;
    }
    // Safe: the quorum check above guarantees n >= 2f + 3 > 2f + 2.
    n - 2 * f - 2
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

    #[test]
    fn test_max_multi_krum_m() {
        // At exactly the quorum floor, only a single vector is selectable --
        // Multi-Krum degenerates to single Krum.
        assert_eq!(max_multi_krum_m(5, 1), 1); // 5 - 2 - 2
        assert_eq!(max_multi_krum_m(6, 1), 2);
        assert_eq!(max_multi_krum_m(7, 1), 3);
        assert_eq!(max_multi_krum_m(7, 2), 1); // quorum floor for f=2
        assert_eq!(max_multi_krum_m(12, 2), 6);
        assert_eq!(max_multi_krum_m(10, 0), 8);
    }

    #[test]
    fn test_max_multi_krum_m_is_zero_below_quorum() {
        assert_eq!(max_multi_krum_m(4, 1), 0);
        assert_eq!(max_multi_krum_m(6, 2), 0);
        assert_eq!(max_multi_krum_m(0, 0), 0);
        // Absurd f from untrusted input must not underflow.
        assert_eq!(max_multi_krum_m(10, usize::MAX), 0);
    }

    #[test]
    fn test_max_multi_krum_m_at_least_one_when_quorum_met() {
        for f in 0..20 {
            let n = krum_min_clients(f);
            assert!(
                max_multi_krum_m(n, f) >= 1,
                "n={} f={} met quorum but reported no safe m",
                n,
                f
            );
        }
    }
}
