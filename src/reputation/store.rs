//! Generic reputation store shared by [`ReputationTracker`](super::ReputationTracker)
//! and [`ByzantineAggregator`](crate::ByzantineAggregator).

use std::borrow::Borrow;
use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};

use super::validate::{validate_adjustment, validate_decay_factor, validate_score};
use crate::error::QoraError;

/// Default trust score for new entries.
pub const DEFAULT_SCORE: f32 = 0.5;

// The default is itself a stored score, so it must satisfy the invariant.
// Checked at compile time because a bad default would corrupt every unknown
// client, every gating decision, and every decay target at once.
const _: () = assert!(DEFAULT_SCORE >= 0.0 && DEFAULT_SCORE <= 1.0);

/// Generic reputation store over any ordered identifier type.
///
/// Wraps a `BTreeMap<ID, f32>` with reputation-specific operations:
/// get with default, reward/penalize with clamping, decay, prune, influence weighting.
///
/// # Invariant
///
/// Every stored score is finite and within `[0.0, 1.0]`. Mutation methods
/// validate their inputs and return [`QoraError`] rather than storing a value
/// that would break it, and deserialization rejects a payload containing one.
/// The read side additionally treats any non-finite score it encounters as
/// untrusted, as defence against state this version did not create.
///
/// Serializes transparently as the underlying map for backward compatibility.
#[derive(Clone, Debug, Serialize)]
#[serde(transparent)]
pub struct ReputationStore<ID: Ord> {
    scores: BTreeMap<ID, f32>,
}

/// Deserialization is a mutation path, so it enforces the same invariant.
///
/// Without this, a persisted file could reintroduce exactly the states the
/// setters now reject -- the validation would hold only for callers who never
/// restore. The whole payload is refused if any single score is invalid;
/// clamping corrupted state silently would discard the evidence that it was
/// corrupt.
impl<'de, ID> Deserialize<'de> for ReputationStore<ID>
where
    ID: Ord + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        let scores = BTreeMap::<ID, f32>::deserialize(deserializer)?;

        for score in scores.values() {
            validate_score(*score).map_err(|_| {
                D::Error::custom(format!(
                    "reputation score {} is outside the storable range [0, 1]",
                    score
                ))
            })?;
        }

        Ok(Self { scores })
    }
}

impl<ID: Ord + Clone> Default for ReputationStore<ID> {
    fn default() -> Self {
        Self::new()
    }
}

impl<ID: Ord + Clone> ReputationStore<ID> {
    /// Create a new, empty reputation store.
    pub fn new() -> Self {
        Self {
            scores: BTreeMap::new(),
        }
    }

    /// Get the score for an identifier (returns [`DEFAULT_SCORE`] for unknown IDs).
    pub fn get_score<Q>(&self, id: &Q) -> f32
    where
        ID: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.scores.get(id).copied().unwrap_or(DEFAULT_SCORE)
    }

    /// Set the score for an identifier.
    ///
    /// # Errors
    ///
    /// [`QoraError::InvalidReputationScore`] if `score` is not finite and
    /// within `[0.0, 1.0]`. Out-of-range values are **rejected, not clamped**:
    /// a caller passing `5.0` has misread the scale, and storing `1.0` would
    /// hide that. The existing score is left untouched on failure.
    pub fn set_score(&mut self, id: ID, score: f32) -> Result<(), QoraError> {
        validate_score(score)?;
        self.scores.insert(id, score);
        Ok(())
    }

    /// Increase a score by `amount`, saturating at 1.0.
    ///
    /// # Errors
    ///
    /// [`QoraError::InvalidReputationAdjustment`] if `amount` is not finite
    /// and non-negative. Amounts above 1.0 are accepted and saturate. The
    /// existing score is left untouched on failure.
    pub fn reward(&mut self, id: ID, amount: f32) -> Result<(), QoraError> {
        validate_adjustment(amount)?;
        let score = self.scores.entry(id).or_insert(DEFAULT_SCORE);
        // Both operands are finite and the sum is clamped, so the invariant
        // holds without relying on min/max's NaN behaviour.
        *score = (*score + amount).clamp(0.0, 1.0);
        Ok(())
    }

    /// Decrease a score by `amount`, saturating at 0.0.
    ///
    /// # Errors
    ///
    /// [`QoraError::InvalidReputationAdjustment`] if `amount` is not finite
    /// and non-negative. Amounts above 1.0 are accepted and saturate. The
    /// existing score is left untouched on failure.
    pub fn penalize(&mut self, id: ID, amount: f32) -> Result<(), QoraError> {
        validate_adjustment(amount)?;
        let score = self.scores.entry(id).or_insert(DEFAULT_SCORE);
        *score = (*score - amount).clamp(0.0, 1.0);
        Ok(())
    }

    /// Check if an ID is banned (score below `ban_threshold`).
    ///
    /// A non-finite stored score counts as banned. Enforcement should make
    /// that unreachable, but the fallback is deliberate: `NaN < threshold` is
    /// false, so a poisoned entry would otherwise be permanently *unbannable*.
    /// Defence in depth for state written by an older version, restored
    /// through a format that admits non-finite floats, or produced by a future
    /// mistake.
    pub fn is_banned<Q>(&self, id: &Q, ban_threshold: f32) -> bool
    where
        ID: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let score = self.get_score(id);
        !score.is_finite() || score < ban_threshold
    }

    /// Compute `min(rep^3, cap)` influence weight.
    pub fn influence_weight<Q>(&self, id: &Q, cap: f32) -> f32
    where
        ID: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let score = self.get_score(id);
        (score * score * score).min(cap)
    }

    /// Move all scores toward [`DEFAULT_SCORE`] by `rate`.
    ///
    /// `0.0` is a no-op and `1.0` sets every score to the default.
    ///
    /// # Errors
    ///
    /// [`QoraError::InvalidReputationDecay`] if `rate` is not finite and
    /// within `[0.0, 1.0]`. The factor is validated **before** the loop, so an
    /// invalid call cannot leave the store partially decayed -- previously a
    /// NaN rate turned every entry into NaN in one pass.
    pub fn decay_toward_default(&mut self, rate: f32) -> Result<(), QoraError> {
        validate_decay_factor(rate)?;

        for score in self.scores.values_mut() {
            // Interpolation between two values in [0, 1] with a factor in
            // [0, 1] stays in range; the clamp absorbs rounding at the edges.
            *score = (*score + rate * (DEFAULT_SCORE - *score)).clamp(0.0, 1.0);
        }
        Ok(())
    }

    /// Remove entries within `epsilon` of [`DEFAULT_SCORE`].
    pub fn prune_near_default(&mut self, epsilon: f32) {
        self.scores
            .retain(|_, score| (*score - DEFAULT_SCORE).abs() > epsilon);
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.scores.clear();
    }

    /// Number of tracked entries.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Iterate over all `(id, score)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&ID, &f32)> {
        self.scores.iter()
    }

    /// Count entries with score below `threshold`.
    ///
    /// Non-finite scores are counted as below it, matching [`Self::is_banned`].
    /// Without this a poisoned entry would be invisible to adaptive trimming,
    /// which uses this count to decide how aggressively to trim.
    pub fn count_below(&self, threshold: f32) -> usize {
        self.scores
            .values()
            .filter(|&&s| !s.is_finite() || s < threshold)
            .count()
    }

    /// Iterate over all scores.
    pub fn scores(&self) -> impl Iterator<Item = &f32> {
        self.scores.values()
    }

    /// Insert a score without validating it.
    ///
    /// Test-only, and deliberately not part of the public API: it exists so
    /// the read-side defences in [`Self::is_banned`] and [`Self::count_below`]
    /// can be exercised against state the validated setters can no longer
    /// produce. Compiled out of non-test builds entirely.
    #[cfg(test)]
    fn insert_corrupted_for_test(&mut self, id: ID, score: f32) {
        self.scores.insert(id, score);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_default_score() {
        let store: ReputationStore<String> = ReputationStore::new();
        assert_eq!(store.get_score("unknown"), DEFAULT_SCORE);
    }

    #[test]
    fn test_store_reward_penalize() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store
            .reward("a".to_string(), 0.1)
            .expect("valid reputation operation");
        assert!((store.get_score("a") - 0.6).abs() < 1e-6);

        store
            .penalize("a".to_string(), 0.3)
            .expect("valid reputation operation");
        assert!((store.get_score("a") - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_store_clamp_bounds() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store
            .reward("a".to_string(), 10.0)
            .expect("valid reputation operation");
        assert_eq!(store.get_score("a"), 1.0);

        store
            .penalize("a".to_string(), 20.0)
            .expect("valid reputation operation");
        assert_eq!(store.get_score("a"), 0.0);
    }

    #[test]
    fn test_store_is_banned() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store
            .penalize("bad".to_string(), 0.4)
            .expect("valid reputation operation");
        assert!(store.is_banned("bad", 0.2));
        assert!(!store.is_banned("unknown", 0.2));
    }

    #[test]
    fn test_store_influence_weight() {
        let store: ReputationStore<String> = ReputationStore::new();
        // Default 0.5: influence = 0.5^3 = 0.125
        assert!((store.influence_weight("x", 0.8) - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_store_influence_cap() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store
            .set_score("a".to_string(), 1.0)
            .expect("valid reputation operation");
        assert!((store.influence_weight("a", 0.8) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_store_decay() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store
            .set_score("a".to_string(), 0.0)
            .expect("valid reputation operation");
        for _ in 0..50 {
            store
                .decay_toward_default(0.1)
                .expect("valid reputation operation");
        }
        assert!((store.get_score("a") - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_store_prune() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store
            .set_score("near_default".to_string(), 0.501)
            .expect("valid reputation operation");
        store
            .set_score("far".to_string(), 0.9)
            .expect("valid reputation operation");
        store.prune_near_default(0.01);
        assert_eq!(store.len(), 1);
        assert!((store.get_score("far") - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_store_serde_string_keys() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store
            .set_score("alice".to_string(), 0.8)
            .expect("valid reputation operation");
        store
            .set_score("bob".to_string(), 0.3)
            .expect("valid reputation operation");

        let json = serde_json::to_string(&store).unwrap();
        let restored: ReputationStore<String> = serde_json::from_str(&json).unwrap();
        assert!((restored.get_score("alice") - 0.8).abs() < 1e-6);
        assert!((restored.get_score("bob") - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_store_count_below() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store
            .set_score("a".to_string(), 0.1)
            .expect("valid reputation operation");
        store
            .set_score("b".to_string(), 0.3)
            .expect("valid reputation operation");
        store
            .set_score("c".to_string(), 0.8)
            .expect("valid reputation operation");
        assert_eq!(store.count_below(0.4), 2);
    }

    // --- Read-side defence against corrupted state ---
    //
    // The validated setters can no longer produce these values. The fallbacks
    // exist for state this version did not write: an older persisted file, a
    // format that admits non-finite floats, or a future mistake. Corruption is
    // injected through a test-only helper rather than a public API.

    fn corrupted(score: f32) -> ReputationStore<String> {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store.insert_corrupted_for_test("poisoned".to_string(), score);
        store
    }

    #[test]
    fn non_finite_scores_are_treated_as_banned() {
        // Every comparison against NaN is false, so `score < threshold` alone
        // would report a poisoned client as *not* banned -- permanently
        // unbannable, which is precisely the pre-fix failure mode.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let store = corrupted(bad);
            assert!(
                store.is_banned("poisoned", 0.2),
                "{} must count as banned",
                bad
            );
        }
    }

    #[test]
    fn non_finite_scores_are_counted_as_below_threshold() {
        // count_below drives adaptive trimming; a poisoned entry that escapes
        // the count would make the store look healthier than it is.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let store = corrupted(bad);
            assert_eq!(store.count_below(0.2), 1, "{} must count as untrusted", bad);
        }
    }

    #[test]
    fn finite_scores_keep_their_ordinary_gating_behavior() {
        // The defensive branch must not change any valid case.
        let mut store: ReputationStore<String> = ReputationStore::new();
        store.set_score("low".to_string(), 0.1).unwrap();
        store.set_score("high".to_string(), 0.9).unwrap();

        assert!(store.is_banned("low", 0.2));
        assert!(!store.is_banned("high", 0.2));
        assert!(!store.is_banned("unknown", 0.2)); // default 0.5
        assert_eq!(store.count_below(0.2), 1);
    }
}
