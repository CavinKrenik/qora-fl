//! Adaptive trimming based on client reputation distribution.
//!
//! Computes a dynamic `trim_fraction` each round based on how many clients
//! have suspicious (low) reputation scores, replacing the fixed trim fraction
//! when enabled on [`ByzantineAggregator`](super::ByzantineAggregator).

/// Compute an adaptive trim fraction from reputation scores.
///
/// Counts what fraction of clients fall below `suspicious_threshold`,
/// then adds `safety_margin` and clamps to `[min_trim, 0.49]`.
///
/// # Arguments
/// * `reputations` - Iterator of client reputation scores
/// * `suspicious_threshold` - Score below which a client is considered suspicious (default 0.4)
/// * `safety_margin` - Extra trim fraction added for safety (default 0.05)
/// * `min_trim` - Minimum trim fraction even when no clients are suspicious (default 0.05)
pub fn compute_adaptive_trim<'a>(
    reputations: impl Iterator<Item = &'a f32>,
    suspicious_threshold: f32,
    safety_margin: f32,
    min_trim: f32,
) -> f32 {
    let scores: Vec<f32> = reputations.copied().collect();
    if scores.is_empty() {
        return min_trim;
    }

    let suspicious_count = scores.iter().filter(|&&s| s < suspicious_threshold).count();
    let suspicious_ratio = suspicious_count as f32 / scores.len() as f32;
    (suspicious_ratio + safety_margin).clamp(min_trim, 0.49)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_suspicious() {
        let scores = vec![0.8, 0.7, 0.9, 0.6, 0.75];
        let trim = compute_adaptive_trim(scores.iter(), 0.4, 0.05, 0.05);
        // 0 suspicious -> 0.0 + 0.05 = 0.05
        assert!((trim - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_all_suspicious() {
        let scores = vec![0.1, 0.2, 0.3, 0.1, 0.15];
        let trim = compute_adaptive_trim(scores.iter(), 0.4, 0.05, 0.05);
        // All 5 suspicious -> 1.0 + 0.05 = 1.05, clamped to 0.49
        assert!((trim - 0.49).abs() < 1e-6);
    }

    #[test]
    fn test_30_percent_suspicious() {
        let scores = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.75];
        let trim = compute_adaptive_trim(scores.iter(), 0.4, 0.05, 0.05);
        // 3/10 suspicious -> 0.30 + 0.05 = 0.35
        assert!((trim - 0.35).abs() < 1e-6);
    }

    #[test]
    fn test_empty() {
        let scores: Vec<f32> = vec![];
        let trim = compute_adaptive_trim(scores.iter(), 0.4, 0.05, 0.05);
        assert!((trim - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_safety_margin_effect() {
        let scores = vec![0.5, 0.6, 0.7, 0.8];
        let trim_low = compute_adaptive_trim(scores.iter(), 0.4, 0.0, 0.0);
        let trim_high = compute_adaptive_trim(scores.iter(), 0.4, 0.2, 0.0);
        assert!(trim_high > trim_low);
    }

    #[test]
    fn test_min_trim_floor() {
        let scores = vec![0.9, 0.9, 0.9];
        let trim = compute_adaptive_trim(scores.iter(), 0.4, 0.0, 0.1);
        // 0 suspicious, margin 0.0, but min_trim 0.1 enforced
        assert!((trim - 0.1).abs() < 1e-6);
    }
}
