//! Reputation Tracking for Sybil Resistance
//!
//! Provides a [`ReputationTracker`] that maintains trust scores per client,
//! backed by the generic [`ReputationStore`](store::ReputationStore).
//!
//! Scores increase with valid contributions and decrease when drift
//! is detected during aggregation. Used to weight clients in
//! Byzantine-tolerant aggregation.
//!
//! # Numeric invariant
//!
//! Every stored score is finite and within `[0.0, 1.0]`, and every operation
//! preserves that:
//!
//! * Scores must be finite and in `[0.0, 1.0]`. Out-of-range values are
//!   rejected rather than clamped.
//! * Reward and penalty amounts must be finite and non-negative. Large finite
//!   amounts are accepted and saturate at the boundary.
//! * Decay factors must be finite and in `[0.0, 1.0]`.
//! * Invalid operations return a typed error and mutate nothing -- not the
//!   targeted entry, and not any other.
//! * Deserialization enforces the same rules, so persisted state cannot
//!   reintroduce a value the setters reject.
//! * Non-finite scores encountered on the read side are treated conservatively
//!   as untrusted (banned, and counted as below any threshold), as defence
//!   against state this version did not write.
//!
//! Distances driving reputation updates are accumulated in `f64` with operands
//! widened before subtraction; see [`crate::math::norms`].
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

pub mod store;
pub(crate) mod validate;

pub use store::ReputationStore;

use serde::{Deserialize, Serialize};

/// A peer identifier (32-byte public key as hex or raw bytes)
pub type PeerId = [u8; 32];

/// Reward increment for valid ZKP submission
const ZKP_REWARD: f32 = 0.02;

/// Penalty for detected drift during aggregation
const DRIFT_PENALTY: f32 = 0.08;

/// Penalty for failed ZKP verification
const ZKP_FAILURE_PENALTY: f32 = 0.15;

/// Ban threshold: peers below this score are excluded
const BAN_THRESHOLD: f32 = 0.2;

/// Maximum influence cap factor for rep^3 weighting (mitigates Slander-Amplification).
const INFLUENCE_CAP: f32 = 0.8;

// Every constant above is fed straight into a validated store operation.
// Checking them at compile time is what lets the wrappers below keep
// infallible signatures: a caller cannot supply these values, so there is no
// runtime failure to report. `0.0 <= x && x <= 1.0` is false for NaN and for
// both infinities, so this also establishes finiteness.
const _: () = assert!(ZKP_REWARD >= 0.0 && ZKP_REWARD <= 1.0);
const _: () = assert!(DRIFT_PENALTY >= 0.0 && DRIFT_PENALTY <= 1.0);
const _: () = assert!(ZKP_FAILURE_PENALTY >= 0.0 && ZKP_FAILURE_PENALTY <= 1.0);
const _: () = assert!(BAN_THRESHOLD >= 0.0 && BAN_THRESHOLD <= 1.0);
const _: () = assert!(INFLUENCE_CAP >= 0.0 && INFLUENCE_CAP <= 1.0);

/// The fixed adjustments [`ReputationTracker`] can apply.
///
/// A closed set of compile-time-validated constants, deliberately not an
/// `f32`: it makes the infallible wrappers below unable to carry a
/// caller-supplied amount into the one place a checked result is discarded.
/// Private to this module and never exposed.
enum ConstAdjustment {
    /// [`ZKP_REWARD`]
    ZkpReward,
    /// [`DRIFT_PENALTY`]
    Drift,
    /// [`ZKP_FAILURE_PENALTY`]
    ZkpFailure,
}

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
    #[serde(flatten)]
    inner: ReputationStore<PeerId>,
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
            inner: ReputationStore::new(),
        }
    }

    /// Get the trust score for a peer (default 0.5 for unknown peers)
    pub fn get_score(&self, peer: &PeerId) -> f32 {
        self.inner.get_score(peer)
    }

    /// Check if a peer is banned (score < BAN_THRESHOLD)
    pub fn is_banned(&self, peer: &PeerId) -> bool {
        self.inner.is_banned(peer, BAN_THRESHOLD)
    }

    /// Reward a peer for submitting a valid ZKP
    ///
    /// Infallible: the amount is the crate constant `ZKP_REWARD`, validated at
    /// compile time, so the underlying checked call cannot fail.
    pub fn reward_valid_zkp(&mut self, peer: &PeerId) {
        self.apply(peer, ConstAdjustment::ZkpReward);
    }

    /// Penalize a peer for drift detected during aggregation
    ///
    /// Infallible for the same reason as [`Self::reward_valid_zkp`].
    pub fn penalize_drift(&mut self, peer: &PeerId) {
        self.apply(peer, ConstAdjustment::Drift);
    }

    /// Penalize a peer for failed ZKP verification
    ///
    /// Infallible for the same reason as [`Self::reward_valid_zkp`].
    pub fn penalize_zkp_failure(&mut self, peer: &PeerId) {
        self.apply(peer, ConstAdjustment::ZkpFailure);
    }

    /// Apply one of the fixed adjustments above.
    ///
    /// The single place in the crate where a checked reputation result is
    /// discarded, so the justification lives in one spot. It takes a
    /// [`ConstAdjustment`] rather than an `f32` or a closure specifically so
    /// that it cannot become an unchecked route for a caller-supplied amount:
    /// the only expressible amounts are the compile-time-validated constants,
    /// and adding another means adding a variant *and* its `const` assertion.
    ///
    /// The `debug_assert` is the backstop for that -- a future constant that
    /// somehow violated the invariant would surface as a test failure rather
    /// than a silently dropped error.
    fn apply(&mut self, peer: &PeerId, adjustment: ConstAdjustment) {
        let result = match adjustment {
            ConstAdjustment::ZkpReward => self.inner.reward(*peer, ZKP_REWARD),
            ConstAdjustment::Drift => self.inner.penalize(*peer, DRIFT_PENALTY),
            ConstAdjustment::ZkpFailure => self.inner.penalize(*peer, ZKP_FAILURE_PENALTY),
        };
        debug_assert!(
            result.is_ok(),
            "reputation constants are compile-time validated: {:?}",
            result
        );
    }

    /// Get all non-banned peers and their scores
    pub fn active_peers(&self) -> Vec<(PeerId, f32)> {
        self.inner
            .iter()
            .filter(|(_, &score)| score >= BAN_THRESHOLD)
            .map(|(&peer, &score)| (peer, score))
            .collect()
    }

    /// Get reputation weights for a set of peers (for weighted aggregation)
    pub fn get_weights(&self, peers: &[PeerId]) -> Vec<f32> {
        peers.iter().map(|p| self.inner.get_score(p)).collect()
    }

    /// Number of tracked peers
    pub fn peer_count(&self) -> usize {
        self.inner.len()
    }

    /// Number of banned peers
    pub fn banned_count(&self) -> usize {
        self.inner.count_below(BAN_THRESHOLD)
    }

    /// Compute the influence-capped reputation^3 weight for a peer.
    ///
    /// Formula: min(rep^3, INFLUENCE_CAP)
    pub fn influence_weight(&self, peer: &PeerId) -> f32 {
        self.inner.influence_weight(peer, INFLUENCE_CAP)
    }

    /// Get influence-capped weights for a set of peers.
    pub fn get_influence_weights(&self, peers: &[PeerId]) -> Vec<f32> {
        peers
            .iter()
            .map(|p| self.inner.influence_weight(p, INFLUENCE_CAP))
            .collect()
    }

    /// Apply reputation decay toward the default trust score (0.5).
    ///
    /// # Arguments
    /// * `rate` - Decay rate in `[0.0, 1.0]`. Typical: 0.01-0.05 per round.
    ///
    /// # Errors
    ///
    /// [`QoraError::InvalidReputationDecay`](crate::QoraError::InvalidReputationDecay)
    /// if `rate` is not finite and within `[0.0, 1.0]`. No score changes when
    /// the rate is rejected.
    pub fn decay_toward_default(&mut self, rate: f32) -> Result<(), crate::error::QoraError> {
        self.inner.decay_toward_default(rate)
    }

    /// Remove peers whose score is within `epsilon` of the default (0.5).
    pub fn prune_default_peers(&mut self, epsilon: f32) {
        self.inner.prune_near_default(epsilon);
    }

    /// Compute influence weight in I16F16-compatible fixed-point representation.
    ///
    /// Returns the influence weight as an i32 in Q16.16 format.
    pub fn influence_weight_fixed(&self, peer: &PeerId) -> i32 {
        let score = self.get_score(peer);
        let rep_cubed = score * score * score;
        let capped = rep_cubed.min(INFLUENCE_CAP);
        let fixed = (capped * 65536.0) as i32;
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
