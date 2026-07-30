//! Reputation numeric safety.
//!
//! One invariant governs this file:
//!
//! > Every stored reputation score is finite and within `[0.0, 1.0]`, and
//! > every arithmetic operation preserves that.
//!
//! Each mutation path previously had a way to break it. `f32::clamp`
//! propagates NaN, so `set_score(id, NaN)` stored NaN and one
//! `decay_toward_default(NaN)` turned the entire store into NaN. `f32::min`
//! and `f32::max` *discard* NaN, so `reward(id, NaN)` produced `1.0` and
//! `penalize(id, NaN)` produced `0.0` -- plausible extremes with no NaN left
//! to notice. Neither clamp constrained an amount's sign, so
//! `reward(id, -5.0)` stored `-4.5`.

use ndarray::{array, Array2};
use qora_fl::aggregators::{AggregationMethod, ByzantineAggregator};
use qora_fl::{QoraError, ReputationStore};

const NON_FINITE: [f32; 3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

fn store() -> ReputationStore<String> {
    ReputationStore::new()
}

fn id(s: &str) -> String {
    s.to_string()
}

// ===== set_score =====

#[test]
fn set_score_accepts_the_closed_unit_interval() {
    let mut s = store();
    for good in [0.0, 0.25, 1.0] {
        s.set_score(id("a"), good).expect("valid score");
        assert_eq!(s.get_score("a"), good);
    }
}

#[test]
fn set_score_rejects_non_finite() {
    for bad in NON_FINITE {
        let mut s = store();
        assert!(
            matches!(
                s.set_score(id("a"), bad),
                Err(QoraError::InvalidReputationScore { .. })
            ),
            "{} must be rejected",
            bad
        );
    }
}

#[test]
fn set_score_rejects_out_of_range_rather_than_clamping() {
    // A caller passing 5.0 has misread the scale; storing 1.0 would hide it.
    for bad in [-0.1, 1.1, 5.0, -1.0] {
        let mut s = store();
        assert!(matches!(
            s.set_score(id("a"), bad),
            Err(QoraError::InvalidReputationScore { .. })
        ));
    }
}

#[test]
fn a_failed_set_score_leaves_the_previous_value_intact() {
    let mut s = store();
    s.set_score(id("a"), 0.7).unwrap();

    for bad in [f32::NAN, f32::INFINITY, -1.0, 2.0] {
        assert!(s.set_score(id("a"), bad).is_err());
        assert_eq!(s.get_score("a"), 0.7, "{} must not have been stored", bad);
    }
}

// ===== reward =====

#[test]
fn reward_increases_and_saturates_at_one() {
    let mut s = store();
    s.reward(id("a"), 0.1).expect("valid amount");
    assert!((s.get_score("a") - 0.6).abs() < 1e-6);

    // Large finite amounts are legal and saturate rather than overshoot.
    s.reward(id("a"), 500.0).expect("large amounts saturate");
    assert_eq!(s.get_score("a"), 1.0);
}

#[test]
fn reward_rejects_non_finite_without_mutating() {
    for bad in NON_FINITE {
        let mut s = store();
        s.set_score(id("a"), 0.4).unwrap();

        assert!(
            matches!(
                s.reward(id("a"), bad),
                Err(QoraError::InvalidReputationAdjustment { .. })
            ),
            "{} must be rejected",
            bad
        );
        // Pre-fix, reward(NaN) silently produced 1.0 -- fully trusted.
        assert_eq!(s.get_score("a"), 0.4);
    }
}

#[test]
fn reward_rejects_negative_amounts() {
    // Pre-fix, reward(id, -5.0) stored -4.5, escaping [0, 1] entirely.
    for bad in [-0.001, -5.0] {
        let mut s = store();
        assert!(matches!(
            s.reward(id("a"), bad),
            Err(QoraError::InvalidReputationAdjustment { .. })
        ));
        assert_eq!(s.get_score("a"), 0.5, "unknown client keeps the default");
    }
}

// ===== penalize =====

#[test]
fn penalize_decreases_and_saturates_at_zero() {
    let mut s = store();
    s.penalize(id("a"), 0.2).expect("valid amount");
    assert!((s.get_score("a") - 0.3).abs() < 1e-6);

    s.penalize(id("a"), 500.0).expect("large amounts saturate");
    assert_eq!(s.get_score("a"), 0.0);
}

#[test]
fn penalize_rejects_non_finite_without_mutating() {
    for bad in NON_FINITE {
        let mut s = store();
        s.set_score(id("a"), 0.4).unwrap();

        assert!(matches!(
            s.penalize(id("a"), bad),
            Err(QoraError::InvalidReputationAdjustment { .. })
        ));
        // Pre-fix, penalize(NaN) silently produced 0.0 -- banned.
        assert_eq!(s.get_score("a"), 0.4);
    }
}

#[test]
fn penalize_rejects_negative_amounts() {
    // Pre-fix, penalize(id, -5.0) stored 5.5 -- above the maximum score.
    for bad in [-0.001, -5.0] {
        let mut s = store();
        assert!(matches!(
            s.penalize(id("a"), bad),
            Err(QoraError::InvalidReputationAdjustment { .. })
        ));
    }
}

#[test]
fn every_valid_adjustment_leaves_the_score_in_range() {
    let mut s = store();
    for amount in [0.0, 0.01, 0.5, 1.0, 3.0, 1e30] {
        s.reward(id("a"), amount).unwrap();
        let score = s.get_score("a");
        assert!(
            score.is_finite() && (0.0..=1.0).contains(&score),
            "reward({}) produced {}",
            amount,
            score
        );

        s.penalize(id("a"), amount).unwrap();
        let score = s.get_score("a");
        assert!(
            score.is_finite() && (0.0..=1.0).contains(&score),
            "penalize({}) produced {}",
            amount,
            score
        );
    }
}

// ===== decay =====

#[test]
fn decay_factor_zero_is_a_no_op_and_one_restores_the_default() {
    let mut s = store();
    s.set_score(id("a"), 0.9).unwrap();

    s.decay_toward_default(0.0).expect("0.0 is valid");
    assert_eq!(s.get_score("a"), 0.9);

    s.decay_toward_default(1.0).expect("1.0 is valid");
    assert!((s.get_score("a") - 0.5).abs() < 1e-6);
}

#[test]
fn fractional_decay_moves_proportionally_toward_the_default() {
    let mut s = store();
    s.set_score(id("a"), 0.9).unwrap();
    s.decay_toward_default(0.5).unwrap();
    // Halfway from 0.9 to 0.5.
    assert!((s.get_score("a") - 0.7).abs() < 1e-6);
}

#[test]
fn decay_rejects_invalid_factors() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
        let mut s = store();
        s.set_score(id("a"), 0.9).unwrap();
        assert!(
            matches!(
                s.decay_toward_default(bad),
                Err(QoraError::InvalidReputationDecay { .. })
            ),
            "{} must be rejected",
            bad
        );
    }
}

#[test]
fn one_invalid_decay_cannot_poison_any_entry() {
    // Pre-fix this was the widest hole in the module: a single NaN factor
    // turned *every* stored score into NaN in one pass. Validating the factor
    // before the loop makes an invalid call atomic.
    let mut s = store();
    for (name, score) in [("a", 0.1), ("b", 0.5), ("c", 0.9)] {
        s.set_score(id(name), score).unwrap();
    }
    let before: Vec<f32> = ["a", "b", "c"].iter().map(|n| s.get_score(*n)).collect();

    assert!(s.decay_toward_default(f32::NAN).is_err());

    let after: Vec<f32> = ["a", "b", "c"].iter().map(|n| s.get_score(*n)).collect();
    assert_eq!(before, after, "no entry may change on a rejected decay");
    assert!(after.iter().all(|s| s.is_finite()));
}

// ===== Deserialization =====

#[test]
fn valid_store_round_trips() {
    let mut s = store();
    s.set_score(id("alice"), 0.8).unwrap();
    s.set_score(id("bob"), 0.0).unwrap();
    s.set_score(id("carol"), 1.0).unwrap();

    let json = serde_json::to_string(&s).expect("serialize");
    let restored: ReputationStore<String> = serde_json::from_str(&json).expect("deserialize");

    for (name, expected) in [("alice", 0.8), ("bob", 0.0), ("carol", 1.0)] {
        assert!((restored.get_score(name) - expected).abs() < 1e-6);
    }
}

#[test]
fn deserialization_rejects_out_of_range_scores() {
    // Without this, a persisted file could reintroduce exactly the states the
    // setters reject, and validation would hold only for callers who never
    // restore.
    for bad in ["-0.1", "1.1", "5.0", "-3.0"] {
        let json = format!(r#"{{"alice":0.5,"mallory":{}}}"#, bad);
        let restored: Result<ReputationStore<String>, _> = serde_json::from_str(&json);
        assert!(
            restored.is_err(),
            "score {} must be refused, not clamped",
            bad
        );
    }
}

#[test]
fn deserialization_accepts_boundary_scores() {
    let json = r#"{"floor":0.0,"ceiling":1.0}"#;
    let restored: ReputationStore<String> =
        serde_json::from_str(json).expect("0.0 and 1.0 are valid");
    assert_eq!(restored.get_score("floor"), 0.0);
    assert_eq!(restored.get_score("ceiling"), 1.0);
}

#[test]
fn deserialization_rejects_the_whole_payload_on_one_bad_score() {
    let json = r#"{"good1":0.5,"bad":2.0,"good2":0.7}"#;
    assert!(
        serde_json::from_str::<ReputationStore<String>>(json).is_err(),
        "a single invalid entry must refuse the payload rather than dropping it"
    );
}

#[test]
fn aggregator_deserialization_rejects_an_invalid_ban_threshold() {
    // `ban_threshold > 0.0` is false for NaN, so a non-finite threshold would
    // silently disable gating on a restored aggregator.
    let json = r#"{"method":"FedAvg","trim_fraction":0.0,"reputation":{},"ban_threshold":-1.0,"adaptive_trim":false}"#;
    assert!(serde_json::from_str::<ByzantineAggregator>(json).is_err());

    let json = r#"{"method":"FedAvg","trim_fraction":0.0,"reputation":{},"ban_threshold":2.0,"adaptive_trim":false}"#;
    assert!(serde_json::from_str::<ByzantineAggregator>(json).is_err());
}

#[test]
fn aggregator_deserialization_accepts_valid_configuration() {
    // The fixtures used by the gating tests must keep working.
    let json = r#"{"method":"FedAvg","trim_fraction":0.0,"reputation":{"a":0.1,"b":0.9},"ban_threshold":0.5,"adaptive_trim":false}"#;
    let agg: ByzantineAggregator = serde_json::from_str(json).expect("valid fixture");
    assert!((agg.get_reputation("a") - 0.1).abs() < 1e-6);
    assert!((agg.ban_threshold() - 0.5).abs() < 1e-6);
}

// ===== Threshold validation =====

#[test]
fn ban_threshold_above_the_default_score_is_legitimate() {
    // A threshold above 0.5 deliberately rejects unknown clients.
    for good in [0.0, 0.2, 0.5, 0.99, 1.0] {
        assert!(
            ByzantineAggregator::with_ban_threshold(AggregationMethod::FedAvg, 0.0, good).is_ok(),
            "{} is a legitimate threshold",
            good
        );
    }
}

#[test]
fn invalid_ban_thresholds_are_rejected_at_construction() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
        assert!(
            matches!(
                ByzantineAggregator::with_ban_threshold(AggregationMethod::FedAvg, 0.0, bad),
                Err(QoraError::InvalidReputationThreshold { .. })
            ),
            "{} must be rejected",
            bad
        );
    }
}

// ===== End-to-end aggregation =====

#[test]
fn aggregation_still_rewards_close_clients_and_penalizes_distant_ones() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);
    let updates = vec![
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[100.0]],
    ];
    let ids: Vec<String> = ["a", "b", "c", "d", "attacker"]
        .iter()
        .map(|s| id(s))
        .collect();

    agg.aggregate(&updates, Some(&ids)).expect("valid round");

    assert!(
        agg.get_reputation("a") > agg.get_reputation("attacker"),
        "close client {} should outrank distant client {}",
        agg.get_reputation("a"),
        agg.get_reputation("attacker")
    );
    for name in ["a", "b", "c", "d", "attacker"] {
        let score = agg.get_reputation(name);
        assert!(
            score.is_finite() && (0.0..=1.0).contains(&score),
            "{} ended at {}",
            name,
            score
        );
    }
}

#[test]
fn aggregation_over_large_finite_updates_produces_a_finite_distance() {
    // The differences here square to ~4e40, past f32::MAX. Computing the
    // distance in f32 would yield `inf`, which now surfaces as
    // NonFiniteReputationDistance -- so a passing round is evidence the
    // operands are widened before subtraction.
    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0);
    let updates: Vec<Array2<f32>> = vec![
        array![[1e20, -1e20]],
        array![[-1e20, 1e20]],
        array![[0.0, 0.0]],
    ];
    let ids: Vec<String> = ["a", "b", "c"].iter().map(|s| id(s)).collect();

    agg.aggregate(&updates, Some(&ids))
        .expect("large finite updates must not produce a non-finite distance");

    for name in ["a", "b", "c"] {
        let score = agg.get_reputation(name);
        assert!(score.is_finite() && (0.0..=1.0).contains(&score));
    }
}

#[test]
fn aggregation_over_tiny_finite_updates_keeps_scores_in_range() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0);
    let updates: Vec<Array2<f32>> = (0..5).map(|i| array![[1e-25 * (i as f32 + 1.0)]]).collect();
    let ids: Vec<String> = (0..5).map(|i| id(&format!("c{}", i))).collect();

    agg.aggregate(&updates, Some(&ids)).expect("valid round");

    for i in 0..5 {
        let score = agg.get_reputation(&format!("c{}", i));
        assert!(score.is_finite() && (0.0..=1.0).contains(&score));
    }
}

#[test]
fn invalid_arithmetic_cannot_make_a_client_unbannable() {
    // The end state of the whole branch: there is no longer a sequence of
    // public calls that leaves a client with a score gating cannot see.
    let mut s = store();
    for bad in NON_FINITE {
        let _ = s.set_score(id("mallory"), bad);
        let _ = s.reward(id("mallory"), bad);
        let _ = s.penalize(id("mallory"), bad);
        let _ = s.decay_toward_default(bad);
    }
    let _ = s.reward(id("mallory"), -1e30);
    let _ = s.penalize(id("mallory"), -1e30);

    let score = s.get_score("mallory");
    assert!(
        score.is_finite() && (0.0..=1.0).contains(&score),
        "score escaped the invariant: {}",
        score
    );

    s.penalize(id("mallory"), 1.0).unwrap();
    assert!(
        s.is_banned("mallory", 0.2),
        "a penalized client must be bannable"
    );
}
