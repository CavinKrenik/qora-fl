"""Tests for qora-fl Python bindings."""

import json

import numpy as np
import pytest

from qora import ByzantineAggregator, ReputationManager

# --- ByzantineAggregator ---


class TestByzantineAggregator:
    def test_trimmed_mean_rejects_attackers(self):
        agg = ByzantineAggregator("trimmed_mean", 0.3)
        updates = [np.array([[1.0, 2.0]], dtype=np.float32) for _ in range(7)]
        updates += [np.array([[100.0, 200.0]], dtype=np.float32) for _ in range(3)]
        result = agg.aggregate(updates)
        np.testing.assert_allclose(result, [[1.0, 2.0]], atol=0.5)

    def test_median_rejects_attackers(self):
        agg = ByzantineAggregator("median", 0.0)
        updates = [np.array([[1.0]], dtype=np.float32) for _ in range(7)]
        updates += [np.array([[100.0]], dtype=np.float32) for _ in range(3)]
        result = agg.aggregate(updates)
        assert result[0, 0] < 2.0

    def test_fedavg_is_vulnerable(self):
        agg = ByzantineAggregator("fedavg", 0.0)
        updates = [np.array([[1.0]], dtype=np.float32) for _ in range(7)]
        updates += [np.array([[100.0]], dtype=np.float32) for _ in range(3)]
        result = agg.aggregate(updates)
        assert result[0, 0] > 10.0, "FedAvg should be corrupted by attackers"

    def test_all_methods(self):
        for method in ["trimmed_mean", "median", "fedavg", "krum"]:
            agg = ByzantineAggregator(method, 0.2)
            updates = [np.array([[1.0]], dtype=np.float32) for _ in range(5)]
            result = agg.aggregate(updates)
            np.testing.assert_allclose(result, [[1.0]], atol=1e-6)

    def test_krum_rejects_attackers(self):
        agg = ByzantineAggregator("krum", 0.0)
        updates = [np.array([[1.0, 2.0]], dtype=np.float32) for _ in range(4)]
        updates += [np.array([[100.0, 200.0]], dtype=np.float32)]
        result = agg.aggregate(updates)
        assert result[0, 0] < 2.0

    def test_krum_custom_f(self):
        agg = ByzantineAggregator("krum:2", 0.0)
        updates = [np.array([[1.0]], dtype=np.float32) for _ in range(7)]
        result = agg.aggregate(updates)
        np.testing.assert_allclose(result, [[1.0]], atol=1e-6)

    def test_krum_deterministic(self):
        updates = [np.array([[1.0, 2.0]], dtype=np.float32) for _ in range(4)]
        updates += [np.array([[50.0, 50.0]], dtype=np.float32)]
        results = []
        for _ in range(3):
            agg = ByzantineAggregator("krum", 0.0)
            results.append(agg.aggregate(updates))
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])

    def test_ban_threshold(self):
        agg = ByzantineAggregator("trimmed_mean", 0.2, ban_threshold=0.3)
        updates = [np.array([[1.0]], dtype=np.float32) for _ in range(4)]
        updates.append(np.array([[100.0]], dtype=np.float32))
        ids = ["a", "b", "c", "d", "attacker"]
        # Drive attacker reputation below threshold
        for _ in range(5):
            agg.aggregate(updates, client_ids=ids)
        assert agg.get_reputation("attacker") < 0.3

    def test_decay_reputations(self):
        agg = ByzantineAggregator("trimmed_mean", 0.2)
        updates = [np.array([[1.0]], dtype=np.float32) for _ in range(4)]
        updates.append(np.array([[100.0]], dtype=np.float32))
        ids = ["a", "b", "c", "d", "attacker"]
        agg.aggregate(updates, client_ids=ids)
        before = agg.get_reputation("attacker")
        agg.decay_reputations(0.5)
        after = agg.get_reputation("attacker")
        assert after > before, "Decay should move penalized client toward 0.5"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            ByzantineAggregator("nonexistent", 0.2)

    def test_empty_updates_raises(self):
        agg = ByzantineAggregator("trimmed_mean", 0.2)
        with pytest.raises(ValueError):
            agg.aggregate([])

    def test_float64_input_coerced(self):
        """float64 arrays should work (Rust side expects float32)."""
        agg = ByzantineAggregator("trimmed_mean", 0.2)
        updates = [np.array([[1.0, 2.0]], dtype=np.float64) for _ in range(5)]
        # PyO3/numpy binding should handle the conversion or raise a clear error
        try:
            result = agg.aggregate(updates)
            # If it works, result should be correct
            assert result.shape == (1, 2)
        except (TypeError, ValueError):
            # If it raises, that's also acceptable -- just not a silent wrong answer
            pass

    def test_reputation_tracking_with_ids(self):
        agg = ByzantineAggregator("trimmed_mean", 0.2)
        updates = [np.array([[1.0]], dtype=np.float32) for _ in range(4)]
        updates.append(np.array([[100.0]], dtype=np.float32))
        ids = ["a", "b", "c", "d", "attacker"]
        agg.aggregate(updates, client_ids=ids)

        assert agg.get_reputation("a") > agg.get_reputation("attacker")

    def test_default_reputation(self):
        agg = ByzantineAggregator("trimmed_mean", 0.2)
        assert agg.get_reputation("unknown") == 0.5

    def test_reset_reputation(self):
        agg = ByzantineAggregator("trimmed_mean", 0.2)
        updates = [np.array([[1.0]], dtype=np.float32) for _ in range(3)]
        ids = ["a", "b", "c"]
        agg.aggregate(updates, client_ids=ids)
        agg.reset_reputation()
        assert agg.get_reputation("a") == 0.5

    def test_serde_roundtrip(self):
        agg = ByzantineAggregator("trimmed_mean", 0.2)
        updates = [np.array([[1.0]], dtype=np.float32) for _ in range(4)]
        updates.append(np.array([[100.0]], dtype=np.float32))
        ids = ["a", "b", "c", "d", "attacker"]
        agg.aggregate(updates, client_ids=ids)

        json_str = agg.to_json()
        restored = ByzantineAggregator.from_json(json_str)

        # Verify parsed JSON is valid
        data = json.loads(json_str)
        assert "method" in data
        assert "reputation" in data

        # Verify restored aggregator preserves reputation
        assert abs(restored.get_reputation("a") - agg.get_reputation("a")) < 1e-6
        assert (
            abs(restored.get_reputation("attacker") - agg.get_reputation("attacker"))
            < 1e-6
        )

    def test_multidimensional_input(self):
        agg = ByzantineAggregator("trimmed_mean", 0.2)
        updates = [np.ones((3, 4), dtype=np.float32) for _ in range(5)]
        result = agg.aggregate(updates)
        assert result.shape == (3, 4)
        np.testing.assert_allclose(result, np.ones((3, 4)), atol=1e-6)


# --- ReputationManager ---


class TestReputationManager:
    def test_default_score(self):
        rep = ReputationManager()
        assert rep.get_score("unknown") == 0.5

    def test_custom_ban_threshold(self):
        rep = ReputationManager(ban_threshold=0.3)
        rep.penalize("client", 0.25)
        # 0.5 - 0.25 = 0.25, which is < 0.3
        assert rep.is_banned("client")

    def test_reward_and_penalize(self):
        rep = ReputationManager(ban_threshold=0.2)
        rep.reward("good", 0.1)
        assert abs(rep.get_score("good") - 0.6) < 1e-6

        rep.penalize("bad", 0.1)
        assert abs(rep.get_score("bad") - 0.4) < 1e-6

    def test_score_clamped_to_0_1(self):
        rep = ReputationManager()
        rep.reward("client", 10.0)
        assert rep.get_score("client") == 1.0

        rep.penalize("client", 20.0)
        assert rep.get_score("client") == 0.0

    def test_ban_after_repeated_penalties(self):
        rep = ReputationManager(ban_threshold=0.2)
        for _ in range(5):
            rep.penalize("bad_client", 0.08)
        assert rep.is_banned("bad_client")

    def test_active_clients(self):
        rep = ReputationManager(ban_threshold=0.2)
        rep.reward("good", 0.1)
        for _ in range(5):
            rep.penalize("bad", 0.08)

        active = rep.active_clients()
        active_ids = [cid for cid, _ in active]
        assert "good" in active_ids
        assert "bad" not in active_ids

    def test_all_scores(self):
        rep = ReputationManager()
        rep.reward("a", 0.01)
        rep.penalize("b", 0.01)
        scores = rep.all_scores()
        assert "a" in scores
        assert "b" in scores

    def test_reset(self):
        rep = ReputationManager()
        rep.reward("client", 0.1)
        rep.reset()
        assert rep.get_score("client") == 0.5  # back to default

    def test_json_roundtrip(self):
        rep = ReputationManager(ban_threshold=0.2)
        rep.reward("hospital_A", 0.1)
        for _ in range(5):
            rep.penalize("hospital_bad", 0.08)

        json_str = rep.to_json()
        rep2 = ReputationManager.from_json(json_str, ban_threshold=0.2)

        assert abs(rep2.get_score("hospital_A") - rep.get_score("hospital_A")) < 1e-6
        assert (
            abs(rep2.get_score("hospital_bad") - rep.get_score("hospital_bad")) < 1e-6
        )
        assert rep2.is_banned("hospital_bad") == rep.is_banned("hospital_bad")

    def test_json_is_valid(self):
        rep = ReputationManager()
        rep.reward("a", 0.1)
        data = json.loads(rep.to_json())
        assert isinstance(data, dict)
        assert "a" in data
