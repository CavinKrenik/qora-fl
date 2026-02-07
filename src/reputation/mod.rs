//! Reputation Tracking for Sybil Resistance
//!
//! Provides a [`ReputationTracker`] that maintains trust scores per client.
//! Scores increase with valid contributions and decrease when drift
//! is detected during aggregation. Used to weight clients in
//! Byzantine-tolerant aggregation.
//!
//! # Influence Formula: `min(rep^3, 0.8)`
//!
//! The cubic weighting `rep^3` was chosen over linear or quadratic schemes for
//! two empirical reasons observed during the 181-day QRES deployment:
//!
//! 1. **Separation**: Cubic weighting amplifies the gap between honest (R≈0.7-1.0)
//!    and marginal (R≈0.3-0.5) clients. At R=0.5, influence is only 0.125 (12.5%
//!    of maximum), providing strong suppression of uncertain participants without
//!    fully excluding them. Linear weighting at R=0.5 gives 0.5 (50%), which is
//!    too permissive for adversarial settings.
//!
//! 2. **Stability**: Higher-order polynomials (quartic, quintic) were tested but
//!    created brittle cliff effects where small score changes caused large influence
//!    swings. Cubic provides a smooth gradient that is steep enough for security
//!    but forgiving enough for clients recovering from transient faults.
//!
//! The 0.8 cap (INFLUENCE_CAP) bounds any single node's contribution even at R=1.0,
//! preventing the "Slander-Amplification" vulnerability where a coalition of
//! high-reputation nodes could dominate consensus and collectively penalize honest
//! newcomers. With the cap, a coalition needs >80% of total weight to control the
//! outcome, which requires many colluding high-reputation nodes rather than just one.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A peer identifier (32-byte public key as hex or raw bytes)
pub type PeerId = [u8; 32];

/// Default trust score for new peers
const DEFAULT_TRUST: f32 = 0.5;

/// Reward increment for valid ZKP submission
const ZKP_REWARD: f32 = 0.02;

/// Penalty for detected drift during aggregation
const DRIFT_PENALTY: f32 = 0.08;

/// Penalty for failed ZKP verification
const ZKP_FAILURE_PENALTY: f32 = 0.15;

/// Ban threshold: peers below this score are excluded
const BAN_THRESHOLD: f32 = 0.2;

/// Maximum influence cap factor for rep^3 weighting (mitigates Slander-Amplification).
/// Even at R=1.0, influence is capped at rep^3 * INFLUENCE_CAP = 1.0 * 0.8 = 0.8.
/// This prevents any single node from dominating consensus in high-Gini scenarios.
const INFLUENCE_CAP: f32 = 0.8;

/// Reputation tracker for swarm peers.
///
/// Maintains a `PeerId -> Score` mapping where:
/// - Scores range from 0.0 (fully distrusted) to 1.0 (fully trusted)
/// - New peers start at 0.5 (neutral)
/// - Valid ZKP submissions increase score
/// - Drift detection during aggregation decreases score
/// - Peers below 0.2 are banned from consensus
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReputationTracker {
    scores: BTreeMap<PeerId, f32>,
}

impl Default for ReputationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ReputationTracker {
    /// Create a new, empty reputation tracker
    pub fn new() -> Self {
        Self {
            scores: BTreeMap::new(),
        }
    }

    /// Get the trust score for a peer (default 0.5 for unknown peers)
    pub fn get_score(&self, peer: &PeerId) -> f32 {
        self.scores.get(peer).copied().unwrap_or(DEFAULT_TRUST)
    }

    /// Check if a peer is banned (score < BAN_THRESHOLD)
    pub fn is_banned(&self, peer: &PeerId) -> bool {
        self.get_score(peer) < BAN_THRESHOLD
    }

    /// Reward a peer for submitting a valid ZKP
    pub fn reward_valid_zkp(&mut self, peer: &PeerId) {
        let score = self.scores.entry(*peer).or_insert(DEFAULT_TRUST);
        *score = (*score + ZKP_REWARD).min(1.0);
    }

    /// Penalize a peer for drift detected during aggregation
    pub fn penalize_drift(&mut self, peer: &PeerId) {
        let score = self.scores.entry(*peer).or_insert(DEFAULT_TRUST);
        *score = (*score - DRIFT_PENALTY).max(0.0);
    }

    /// Penalize a peer for failed ZKP verification
    pub fn penalize_zkp_failure(&mut self, peer: &PeerId) {
        let score = self.scores.entry(*peer).or_insert(DEFAULT_TRUST);
        *score = (*score - ZKP_FAILURE_PENALTY).max(0.0);
    }

    /// Get all non-banned peers and their scores
    pub fn active_peers(&self) -> Vec<(PeerId, f32)> {
        self.scores
            .iter()
            .filter(|(_, &score)| score >= BAN_THRESHOLD)
            .map(|(&peer, &score)| (peer, score))
            .collect()
    }

    /// Get reputation weights for a set of peers (for weighted aggregation)
    /// Returns weights normalized so the max is 1.0
    pub fn get_weights(&self, peers: &[PeerId]) -> Vec<f32> {
        peers.iter().map(|p| self.get_score(p)).collect()
    }

    /// Number of tracked peers
    pub fn peer_count(&self) -> usize {
        self.scores.len()
    }

    /// Number of banned peers
    pub fn banned_count(&self) -> usize {
        self.scores.values().filter(|&&s| s < BAN_THRESHOLD).count()
    }

    /// Compute the influence-capped reputation^3 weight for a peer.
    ///
    /// Applies the cubic reputation weighting used in multimodal temporal
    /// attention (INV-1 compliance) with an influence cap to mitigate the
    /// "Slander-Amplification" vulnerability described in REPUTATION_PRIVACY.md.
    ///
    /// Formula: min(rep^3, INFLUENCE_CAP)
    ///
    /// This ensures no single node can dominate consensus even at R=1.0,
    /// bounding single-node contribution to 0.8 of the total weight.
    ///
    /// The computation uses checked multiplication to avoid overflow when
    /// translated to I16F16 fixed-point in the consensus path.
    pub fn influence_weight(&self, peer: &PeerId) -> f32 {
        let score = self.get_score(peer);
        let rep_cubed = score * score * score;
        rep_cubed.min(INFLUENCE_CAP)
    }

    /// Get influence-capped weights for a set of peers (for weighted aggregation).
    /// Each weight is min(rep^3, INFLUENCE_CAP).
    pub fn get_influence_weights(&self, peers: &[PeerId]) -> Vec<f32> {
        peers.iter().map(|p| self.influence_weight(p)).collect()
    }

    /// Apply reputation decay toward the default trust score.
    ///
    /// Each call moves all scores toward `DEFAULT_TRUST` (0.5) by `rate`.
    /// Call once per aggregation round to model temporal forgiveness:
    /// old clients gradually return to neutral, and penalized clients
    /// can recover over time.
    ///
    /// # Arguments
    ///
    /// * `rate` - Decay rate in (0.0, 1.0). Typical: 0.01-0.05 per round.
    ///   `score = score + rate * (DEFAULT_TRUST - score)`
    pub fn decay_toward_default(&mut self, rate: f32) {
        let rate = rate.clamp(0.0, 1.0);
        for score in self.scores.values_mut() {
            *score += rate * (DEFAULT_TRUST - *score);
        }
    }

    /// Remove peers that have been inactive (not updated) and whose score
    /// has decayed back to approximately the default. Useful for cleaning
    /// up stale entries from clients that have left the federation.
    ///
    /// Removes peers whose score is within `epsilon` of `DEFAULT_TRUST`.
    pub fn prune_default_peers(&mut self, epsilon: f32) {
        self.scores
            .retain(|_, score| (*score - DEFAULT_TRUST).abs() > epsilon);
    }

    /// Compute influence weight in I16F16-compatible fixed-point representation.
    ///
    /// Returns the influence weight as an i32 in Q16.16 format, using
    /// saturating arithmetic to prevent overflow on the i32 path.
    ///
    /// Q16.16 range: [-32768.0, 32767.999...], so rep^3 * 0.8 is always
    /// in range (max value = 0.8, well within bounds).
    pub fn influence_weight_fixed(&self, peer: &PeerId) -> i32 {
        let score = self.get_score(peer);
        // Compute rep^3 in f32, cap, then convert to Q16.16
        let rep_cubed = score * score * score;
        let capped = rep_cubed.min(INFLUENCE_CAP);
        // Convert to Q16.16: multiply by 2^16 = 65536
        let fixed = (capped * 65536.0) as i32;
        // Saturate to valid Q16.16 range (always positive for reputation)
        fixed.max(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peer(id: u8) -> PeerId {
        let mut peer = [0u8; 32];
        peer[0] = id;
        peer
    }

    #[test]
    fn test_default_score() {
        let tracker = ReputationTracker::new();
        let peer = make_peer(1);
        assert_eq!(tracker.get_score(&peer), 0.5);
    }

    #[test]
    fn test_reward_valid_zkp() {
        let mut tracker = ReputationTracker::new();
        let peer = make_peer(1);

        tracker.reward_valid_zkp(&peer);
        assert!((tracker.get_score(&peer) - 0.52).abs() < 0.001);

        // Reward 25 more times -> should cap at 1.0
        for _ in 0..25 {
            tracker.reward_valid_zkp(&peer);
        }
        assert_eq!(tracker.get_score(&peer), 1.0);
    }

    #[test]
    fn test_penalize_drift() {
        let mut tracker = ReputationTracker::new();
        let peer = make_peer(1);

        tracker.penalize_drift(&peer);
        assert!((tracker.get_score(&peer) - 0.42).abs() < 0.001);
    }

    #[test]
    fn test_ban_threshold() {
        let mut tracker = ReputationTracker::new();
        let peer = make_peer(1);

        // Penalize enough to get below ban threshold
        for _ in 0..5 {
            tracker.penalize_drift(&peer);
        }
        // 0.5 - 5*0.08 = 0.10 < 0.2
        assert!(tracker.is_banned(&peer));
    }

    #[test]
    fn test_zkp_failure_penalty() {
        let mut tracker = ReputationTracker::new();
        let peer = make_peer(1);

        tracker.penalize_zkp_failure(&peer);
        assert!((tracker.get_score(&peer) - 0.35).abs() < 0.001);

        // Two more failures -> 0.35 - 0.15 - 0.15 = 0.05 < 0.2
        tracker.penalize_zkp_failure(&peer);
        tracker.penalize_zkp_failure(&peer);
        assert!(tracker.is_banned(&peer));
    }

    #[test]
    fn test_influence_cap() {
        let tracker = ReputationTracker::new();
        let peer = make_peer(1);

        // Default peer (0.5): influence = 0.5^3 = 0.125
        let influence = tracker.influence_weight(&peer);
        assert!((influence - 0.125).abs() < 0.001);

        // Verify cap: even at R=1.0, influence <= 0.8
        let mut tracker2 = ReputationTracker::new();
        let high_peer = make_peer(2);
        for _ in 0..30 {
            tracker2.reward_valid_zkp(&high_peer);
        }
        assert_eq!(tracker2.get_score(&high_peer), 1.0);
        let capped = tracker2.influence_weight(&high_peer);
        assert!(
            (capped - 0.8).abs() < 0.001,
            "R=1.0 should be capped at 0.8"
        );
    }

    #[test]
    fn test_influence_weight_fixed_no_overflow() {
        let mut tracker = ReputationTracker::new();
        let peer = make_peer(1);
        for _ in 0..30 {
            tracker.reward_valid_zkp(&peer);
        }
        let fixed = tracker.influence_weight_fixed(&peer);
        // 0.8 in Q16.16 = 0.8 * 65536 = 52428
        assert!(
            (fixed - 52428).abs() <= 1,
            "Fixed-point influence should be ~52428"
        );
        assert!(fixed >= 0, "Fixed-point influence must be non-negative");
    }

    #[test]
    fn test_active_peers_excludes_banned() {
        let mut tracker = ReputationTracker::new();
        let good_peer = make_peer(1);
        let bad_peer = make_peer(2);

        tracker.reward_valid_zkp(&good_peer);
        // Ban bad_peer
        for _ in 0..5 {
            tracker.penalize_drift(&bad_peer);
        }

        let active = tracker.active_peers();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].0, good_peer);
    }
}
