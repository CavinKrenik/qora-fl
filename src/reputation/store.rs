//! Generic reputation store shared by [`ReputationTracker`](super::ReputationTracker)
//! and [`ByzantineAggregator`](crate::ByzantineAggregator).

use std::borrow::Borrow;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Default trust score for new entries.
pub const DEFAULT_SCORE: f32 = 0.5;

/// Generic reputation store over any ordered identifier type.
///
/// Wraps a `BTreeMap<ID, f32>` with reputation-specific operations:
/// get with default, reward/penalize with clamping, decay, prune, influence weighting.
///
/// Serializes transparently as the underlying map for backward compatibility.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReputationStore<ID: Ord> {
    scores: BTreeMap<ID, f32>,
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

    /// Set the score for an identifier (clamped to \[0.0, 1.0\]).
    pub fn set_score(&mut self, id: ID, score: f32) {
        self.scores.insert(id, score.clamp(0.0, 1.0));
    }

    /// Increase a score by `amount` (capped at 1.0).
    pub fn reward(&mut self, id: ID, amount: f32) {
        let score = self.scores.entry(id).or_insert(DEFAULT_SCORE);
        *score = (*score + amount).min(1.0);
    }

    /// Decrease a score by `amount` (floored at 0.0).
    pub fn penalize(&mut self, id: ID, amount: f32) {
        let score = self.scores.entry(id).or_insert(DEFAULT_SCORE);
        *score = (*score - amount).max(0.0);
    }

    /// Check if an ID is banned (score below `ban_threshold`).
    pub fn is_banned<Q>(&self, id: &Q, ban_threshold: f32) -> bool
    where
        ID: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get_score(id) < ban_threshold
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
    pub fn decay_toward_default(&mut self, rate: f32) {
        let rate = rate.clamp(0.0, 1.0);
        for score in self.scores.values_mut() {
            *score += rate * (DEFAULT_SCORE - *score);
        }
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
    pub fn count_below(&self, threshold: f32) -> usize {
        self.scores.values().filter(|&&s| s < threshold).count()
    }

    /// Iterate over all scores.
    pub fn scores(&self) -> impl Iterator<Item = &f32> {
        self.scores.values()
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
        store.reward("a".to_string(), 0.1);
        assert!((store.get_score("a") - 0.6).abs() < 1e-6);

        store.penalize("a".to_string(), 0.3);
        assert!((store.get_score("a") - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_store_clamp_bounds() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store.reward("a".to_string(), 10.0);
        assert_eq!(store.get_score("a"), 1.0);

        store.penalize("a".to_string(), 20.0);
        assert_eq!(store.get_score("a"), 0.0);
    }

    #[test]
    fn test_store_is_banned() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store.penalize("bad".to_string(), 0.4);
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
        store.set_score("a".to_string(), 1.0);
        assert!((store.influence_weight("a", 0.8) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_store_decay() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store.set_score("a".to_string(), 0.0);
        for _ in 0..50 {
            store.decay_toward_default(0.1);
        }
        assert!((store.get_score("a") - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_store_prune() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store.set_score("near_default".to_string(), 0.501);
        store.set_score("far".to_string(), 0.9);
        store.prune_near_default(0.01);
        assert_eq!(store.len(), 1);
        assert!((store.get_score("far") - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_store_serde_string_keys() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store.set_score("alice".to_string(), 0.8);
        store.set_score("bob".to_string(), 0.3);

        let json = serde_json::to_string(&store).unwrap();
        let restored: ReputationStore<String> = serde_json::from_str(&json).unwrap();
        assert!((restored.get_score("alice") - 0.8).abs() < 1e-6);
        assert!((restored.get_score("bob") - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_store_count_below() {
        let mut store: ReputationStore<String> = ReputationStore::new();
        store.set_score("a".to_string(), 0.1);
        store.set_score("b".to_string(), 0.3);
        store.set_score("c".to_string(), 0.8);
        assert_eq!(store.count_below(0.4), 2);
    }
}
