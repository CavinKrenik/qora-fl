//! Integration tests for Qora-FL aggregation algorithms

use ndarray::array;
use ndarray::Array2;
use qora_fl::aggregators::krum::aggregate_krum;
use qora_fl::aggregators::{AggregationMethod, ByzantineAggregator};
use qora_fl::error::QoraError;
use qora_fl::reputation::ReputationTracker;
use qora_fl::{fedavg, median, trimmed_mean};

use fixed::types::I16F16;

#[test]
fn test_trimmed_mean_30_percent_attack() {
    // 7 honest + 3 Byzantine (30%)
    let mut updates = vec![array![[1.0]]; 7];
    updates.extend(vec![array![[100.0]]; 3]);

    let result = trimmed_mean(&updates, 0.3).unwrap();

    // Should be close to honest mean (1.0)
    assert!(
        (result[[0, 0]] - 1.0).abs() < 0.5,
        "Trimmed mean should resist 30% attack, got {}",
        result[[0, 0]]
    );
}

#[test]
fn test_median_robust() {
    let updates = vec![
        array![[1.0]],
        array![[2.0]],
        array![[100.0]], // Outlier
    ];

    let result = median(&updates).unwrap();
    assert_eq!(result[[0, 0]], 2.0);
}

#[test]
fn test_fedavg_vulnerable() {
    let updates = vec![
        array![[1.0]],
        array![[1.0]],
        array![[100.0]], // Single attacker poisons result
    ];

    let result = fedavg(&updates, None).unwrap();

    // FedAvg is corrupted
    assert!(
        result[[0, 0]] > 10.0,
        "FedAvg should be vulnerable, got {}",
        result[[0, 0]]
    );
}

#[test]
fn test_byzantine_aggregator_workflow() {
    // trim_fraction=0.3 trims ceil(10*0.3)=3 from each end, covering 30% attackers
    let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.3);

    // Simulate 3 FL rounds
    for round in 0..3 {
        let mut updates = vec![array![[1.0 + round as f32 * 0.1]]; 7];
        updates.extend(vec![array![[100.0]]; 3]); // 30% attackers

        let ids: Vec<String> = (0..10).map(|i| format!("client_{}", i)).collect();
        let result = agg.aggregate(&updates, Some(&ids)).unwrap();

        // Should always be close to honest values
        assert!(
            (result[[0, 0]] - (1.0 + round as f32 * 0.1)).abs() < 0.5,
            "Round {}: expected ~{}, got {}",
            round,
            1.0 + round as f32 * 0.1,
            result[[0, 0]]
        );
    }

    // After 3 rounds, honest clients should have better reputation
    let honest_rep = agg.get_reputation("client_0");
    let byzantine_rep = agg.get_reputation("client_9");
    assert!(
        honest_rep > byzantine_rep,
        "Honest ({}) should have better reputation than Byzantine ({})",
        honest_rep,
        byzantine_rep
    );
}

#[test]
fn test_all_methods_agree_on_honest_data() {
    // When all clients are honest, all methods should give similar results
    let updates = vec![
        array![[1.0, 2.0]],
        array![[1.1, 2.1]],
        array![[0.9, 1.9]],
        array![[1.0, 2.0]],
        array![[1.05, 1.95]],
    ];

    let tm = trimmed_mean(&updates, 0.2).unwrap();
    let med = median(&updates).unwrap();
    let avg = fedavg(&updates, None).unwrap();

    // Krum via ByzantineAggregator (selects one vector, so it's one of the inputs)
    let mut krum_agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);
    let krum = krum_agg.aggregate(&updates, None).unwrap();

    // All should be close to [1.0, 2.0]
    for result in &[&tm, &med, &avg, &krum] {
        assert!(
            (result[[0, 0]] - 1.0).abs() < 0.15,
            "Method disagrees on param 0: {}",
            result[[0, 0]]
        );
        assert!(
            (result[[0, 1]] - 2.0).abs() < 0.15,
            "Method disagrees on param 1: {}",
            result[[0, 1]]
        );
    }
}

#[test]
fn test_large_model_update() {
    // Test with larger arrays (simulating real model parameters)
    let n_clients = 20;
    let n_byzantine = 6; // 30%
    let n_honest = n_clients - n_byzantine;
    let rows = 10;
    let cols = 10;

    let mut updates: Vec<Array2<f32>> = Vec::new();

    // Honest clients: values near 0.5
    for _ in 0..n_honest {
        updates.push(Array2::from_elem((rows, cols), 0.5));
    }

    // Byzantine clients: values at 1000.0
    for _ in 0..n_byzantine {
        updates.push(Array2::from_elem((rows, cols), 1000.0));
    }

    let result = trimmed_mean(&updates, 0.3).unwrap();

    // Every coordinate should be close to 0.5
    for row in 0..rows {
        for col in 0..cols {
            assert!(
                (result[[row, col]] - 0.5).abs() < 0.1,
                "Large model: [{},{}] = {}, expected ~0.5",
                row,
                col,
                result[[row, col]]
            );
        }
    }
}

#[test]
fn test_comparison_trimmed_mean_vs_fedavg() {
    // Demonstrate that trimmed mean resists attacks while FedAvg doesn't
    let mut updates = vec![array![[1.0]]; 7];
    updates.extend(vec![array![[100.0]]; 3]);

    let robust = trimmed_mean(&updates, 0.3).unwrap();
    let naive = fedavg(&updates, None).unwrap();

    // Trimmed mean stays close to honest value
    assert!(
        robust[[0, 0]] < 2.0,
        "Trimmed mean should resist: {}",
        robust[[0, 0]]
    );
    // FedAvg gets corrupted
    assert!(
        naive[[0, 0]] > 20.0,
        "FedAvg should be corrupted: {}",
        naive[[0, 0]]
    );
}

// ===== Phase 2: Critical missing tests =====

// --- Median at 50% Byzantine tolerance ---

#[test]
fn test_median_at_exactly_50_percent_boundary() {
    // 5 honest + 5 Byzantine (exactly 50%) -- median is on the boundary
    let mut updates = vec![array![[3.0]]; 5];
    updates.extend(vec![array![[999.0]]; 5]);

    let result = median(&updates).unwrap();
    // Sorted: [3,3,3,3,3,999,999,999,999,999]
    // Even count: median = average of 5th and 6th = (3+999)/2 = 501
    // At exactly 50%, the median is compromised. This is expected —
    // median tolerates *strictly less than* 50%.
    assert!(
        (result[[0, 0]] - 501.0).abs() < 1e-6,
        "At exactly 50%, median averages boundary values, got {}",
        result[[0, 0]]
    );
}

#[test]
fn test_median_below_50_percent_byzantine() {
    // 6 honest + 4 Byzantine (<50%) -- median equals honest value exactly
    let mut updates = vec![array![[5.0]]; 6];
    updates.extend(vec![array![[999.0]]; 4]);

    let result = median(&updates).unwrap();
    // Sorted: [5,5,5,5,5,5,999,999,999,999] -> median = (5+5)/2 = 5.0
    assert!(
        (result[[0, 0]] - 5.0).abs() < 1e-6,
        "Median with <50% Byzantine should equal honest value, got {}",
        result[[0, 0]]
    );
}

// --- Single client edge cases ---

#[test]
fn test_trimmed_mean_single_client_zero_trim() {
    // With trim_fraction=0.0, single client should return its own update
    let updates = vec![array![[42.0, 7.0]]];
    let result = trimmed_mean(&updates, 0.0).unwrap();
    assert!(
        (result[[0, 0]] - 42.0).abs() < 1e-6,
        "Single client trimmed_mean(0.0) should return that client's update"
    );
}

#[test]
fn test_trimmed_mean_single_client_nonzero_trim_errors() {
    // With trim_fraction > 0 and only 1 client, trimming leaves 0 values -> error
    let updates = vec![array![[42.0, 7.0]]];
    let result = trimmed_mean(&updates, 0.2);
    assert!(
        result.is_err(),
        "Single client with non-zero trim should error"
    );
    assert!(matches!(
        result.unwrap_err(),
        QoraError::InsufficientQuorum { .. }
    ));
}

#[test]
fn test_median_single_client() {
    let updates = vec![array![[42.0, 7.0]]];
    let result = median(&updates).unwrap();
    assert!(
        (result[[0, 0]] - 42.0).abs() < 1e-6,
        "Single client median should return that client's update"
    );
}

// --- Error path coverage ---

#[test]
fn test_trimmed_mean_empty_returns_error() {
    let updates: Vec<Array2<f32>> = vec![];
    let result = trimmed_mean(&updates, 0.2);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QoraError::EmptyUpdates));
}

#[test]
fn test_median_empty_returns_error() {
    let updates: Vec<Array2<f32>> = vec![];
    let result = median(&updates);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QoraError::EmptyUpdates));
}

#[test]
fn test_fedavg_empty_returns_error() {
    let updates: Vec<Array2<f32>> = vec![];
    let result = fedavg(&updates, None);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QoraError::EmptyUpdates));
}

#[test]
fn test_trimmed_mean_dimension_mismatch() {
    let updates = vec![array![[1.0, 2.0]], array![[1.0]]];
    let result = trimmed_mean(&updates, 0.2);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QoraError::DimensionMismatch));
}

#[test]
fn test_trimmed_mean_invalid_fraction_high() {
    let updates = vec![array![[1.0]], array![[2.0]]];
    let result = trimmed_mean(&updates, 0.6);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QoraError::InvalidTrimFraction(_)
    ));
}

#[test]
fn test_trimmed_mean_invalid_fraction_negative() {
    let updates = vec![array![[1.0]], array![[2.0]]];
    let result = trimmed_mean(&updates, -0.1);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QoraError::InvalidTrimFraction(_)
    ));
}

#[test]
fn test_fedavg_dimension_mismatch() {
    let updates = vec![array![[1.0, 2.0]], array![[1.0]]];
    let result = fedavg(&updates, None);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QoraError::DimensionMismatch));
}

#[test]
fn test_fedavg_weight_count_mismatch() {
    let updates = vec![array![[1.0]], array![[2.0]]];
    let result = fedavg(&updates, Some(&[1.0])); // 1 weight for 2 updates
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QoraError::DimensionMismatch));
}

#[test]
fn test_fedavg_zero_weights() {
    let updates = vec![array![[1.0]], array![[2.0]]];
    let result = fedavg(&updates, Some(&[0.0, 0.0]));
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QoraError::InsufficientQuorum { .. }
    ));
}

#[test]
fn test_error_display_impls() {
    // Exercise Display for all QoraError variants (covers error.rs)
    let e = QoraError::EmptyUpdates;
    assert_eq!(format!("{}", e), "Empty updates provided");

    let e = QoraError::DimensionMismatch;
    assert_eq!(format!("{}", e), "Dimension mismatch in updates");

    let e = QoraError::InvalidTrimFraction(0.6);
    assert!(format!("{}", e).contains("0.6"));

    let e = QoraError::InsufficientQuorum {
        needed: 5,
        actual: 2,
    };
    let msg = format!("{}", e);
    assert!(msg.contains("5") && msg.contains("2"));

    let e = QoraError::ReputationError("test".to_string());
    assert!(format!("{}", e).contains("test"));

    let e = QoraError::VerificationError("verify".to_string());
    assert!(format!("{}", e).contains("verify"));

    let e = QoraError::ShapeError("shape".to_string());
    assert!(format!("{}", e).contains("shape"));
}

// --- Krum at exact n=2f+3 boundary ---

#[test]
fn test_krum_at_exact_boundary() {
    // n = 2f+3 = 2*1+3 = 5, f=1
    let honest = vec![I16F16::from_num(1.0), I16F16::from_num(1.0)];
    let vectors = vec![
        honest.clone(),
        vec![I16F16::from_num(1.1), I16F16::from_num(0.9)],
        vec![I16F16::from_num(0.9), I16F16::from_num(1.1)],
        vec![I16F16::from_num(1.05), I16F16::from_num(0.95)],
        vec![I16F16::from_num(100.0), I16F16::from_num(100.0)], // Byzantine
    ];
    let result = aggregate_krum(&vectors, 1).expect("Should succeed at exact boundary");
    assert!(
        result[0] < I16F16::from_num(2.0),
        "Krum at exact n=2f+3 should select honest vector"
    );
}

#[test]
fn test_krum_below_2f3_best_effort() {
    // n=4, f=2 -> 2f+3=7, so n < 2f+3. Should still return a result (best-effort).
    let vectors = vec![
        vec![I16F16::from_num(1.0)],
        vec![I16F16::from_num(1.1)],
        vec![I16F16::from_num(100.0)],
        vec![I16F16::from_num(100.0)],
    ];
    // Should return Some (best-effort), not None
    let result = aggregate_krum(&vectors, 2);
    assert!(
        result.is_some(),
        "Krum below 2f+3 should still return best-effort result"
    );
}

// --- BFP16 round-trip ---

#[test]
fn test_bfp16_round_trip_accuracy() {
    use qora_fl::aggregators::krum::Bfp16Vec;

    // Values in a similar magnitude range (shared exponent means dynamic range is limited)
    let original = vec![1.0f32, -0.5, 0.25, 2.0, -1.5];
    let bfp = Bfp16Vec::from_f32_slice(&original);
    let recovered = bfp.to_vec_f32();

    for (o, r) in original.iter().zip(recovered.iter()) {
        let rel_error = if o.abs() > 1e-6 {
            (o - r).abs() / o.abs()
        } else {
            (o - r).abs()
        };
        assert!(
            rel_error < 0.01,
            "BFP16 round-trip: {} -> {}, rel_error={}",
            o,
            r,
            rel_error
        );
    }
}

#[test]
fn test_bfp16_empty_and_zeros() {
    use qora_fl::aggregators::krum::Bfp16Vec;

    let empty = Bfp16Vec::from_f32_slice(&[]);
    assert!(empty.mantissas.is_empty());

    let zeros = Bfp16Vec::from_f32_slice(&[0.0, 0.0, 0.0]);
    let recovered = zeros.to_vec_f32();
    assert!(recovered.iter().all(|&x| x == 0.0));
}

// --- ReputationTracker serialization round-trip ---

#[test]
fn test_reputation_tracker_default_impl() {
    // ReputationTracker implements Default (same as ::new())
    let tracker = ReputationTracker::default();
    let peer = {
        let mut p = [0u8; 32];
        p[0] = 1;
        p
    };
    assert_eq!(tracker.get_score(&peer), 0.5);
    assert!(!tracker.is_banned(&peer));
}

#[test]
fn test_reputation_tracker_score_boundaries() {
    let mut tracker = ReputationTracker::new();
    let peer = {
        let mut p = [0u8; 32];
        p[0] = 1;
        p
    };

    // Reward many times -> should cap at 1.0
    for _ in 0..100 {
        tracker.reward_valid_zkp(&peer);
    }
    assert_eq!(tracker.get_score(&peer), 1.0);

    // Penalize many times -> should floor at 0.0
    for _ in 0..200 {
        tracker.penalize_drift(&peer);
    }
    assert_eq!(tracker.get_score(&peer), 0.0);
    assert!(tracker.is_banned(&peer));
}

// --- ByzantineAggregator serialization round-trip ---

#[test]
fn test_byzantine_aggregator_serde_roundtrip() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);

    let updates = vec![array![[1.0]], array![[1.0]], array![[1.0]], array![[100.0]]];
    let ids = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
    ];
    let _ = agg.aggregate(&updates, Some(&ids)).unwrap();

    let json = serde_json::to_string(&agg).expect("serialize");
    let restored: ByzantineAggregator = serde_json::from_str(&json).expect("deserialize");

    assert!(
        (restored.get_reputation("a") - agg.get_reputation("a")).abs() < 1e-6,
        "Reputation mismatch after round-trip"
    );
    assert!(
        (restored.get_reputation("d") - agg.get_reputation("d")).abs() < 1e-6,
        "Byzantine client reputation mismatch after round-trip"
    );
}

// --- ReputationTracker utility functions ---

#[test]
fn test_reputation_tracker_peer_count_and_banned_count() {
    let mut tracker = ReputationTracker::new();
    assert_eq!(tracker.peer_count(), 0);
    assert_eq!(tracker.banned_count(), 0);

    let peer1 = {
        let mut p = [0u8; 32];
        p[0] = 1;
        p
    };
    let peer2 = {
        let mut p = [0u8; 32];
        p[0] = 2;
        p
    };

    tracker.reward_valid_zkp(&peer1);
    assert_eq!(tracker.peer_count(), 1);
    assert_eq!(tracker.banned_count(), 0);

    for _ in 0..7 {
        tracker.penalize_drift(&peer2);
    }
    assert_eq!(tracker.peer_count(), 2);
    assert_eq!(tracker.banned_count(), 1);
}

#[test]
fn test_reputation_tracker_get_weights() {
    let mut tracker = ReputationTracker::new();
    let peer1 = {
        let mut p = [0u8; 32];
        p[0] = 1;
        p
    };
    let peer2 = {
        let mut p = [0u8; 32];
        p[0] = 2;
        p
    };

    tracker.reward_valid_zkp(&peer1);
    tracker.penalize_drift(&peer2);

    let weights = tracker.get_weights(&[peer1, peer2]);
    assert_eq!(weights.len(), 2);
    assert!((weights[0] - 0.52).abs() < 0.001);
    assert!((weights[1] - 0.42).abs() < 0.001);

    let influence_weights = tracker.get_influence_weights(&[peer1, peer2]);
    assert_eq!(influence_weights.len(), 2);
    // peer1: 0.52^3 = 0.140608
    assert!((influence_weights[0] - 0.52f32.powi(3)).abs() < 0.001);
}

// --- Reputation influence formula at 0.8 cap ---

#[test]
fn test_influence_cap_at_0_8() {
    let mut tracker = ReputationTracker::new();
    let peer = {
        let mut p = [0u8; 32];
        p[0] = 99;
        p
    };

    // Boost to max reputation
    for _ in 0..30 {
        tracker.reward_valid_zkp(&peer);
    }
    assert_eq!(tracker.get_score(&peer), 1.0);

    let influence = tracker.influence_weight(&peer);
    assert!(
        (influence - 0.8).abs() < 1e-6,
        "Influence at R=1.0 must be exactly 0.8, got {}",
        influence
    );

    // Also verify the fixed-point version
    let fixed = tracker.influence_weight_fixed(&peer);
    // 0.8 * 65536 = 52428.8 -> 52428
    assert!(
        (fixed - 52428).abs() <= 1,
        "Fixed-point influence at R=1.0 should be ~52428, got {}",
        fixed
    );
}

// ===== Phase 3: Krum via ByzantineAggregator + Determinism =====

#[test]
fn test_krum_via_byzantine_aggregator() {
    // Test Krum through the high-level API (f32 <-> I16F16 bridge)
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);

    let updates = vec![
        array![[1.0, 2.0]],
        array![[1.1, 2.1]],
        array![[0.9, 1.9]],
        array![[1.05, 2.05]],
        array![[50.0, 50.0]], // Byzantine
    ];

    let result = agg.aggregate(&updates, None).unwrap();
    // Should select an honest vector, not the Byzantine one
    assert!(
        result[[0, 0]] < 2.0,
        "Krum via ByzantineAggregator should select honest vector, got {}",
        result[[0, 0]]
    );
}

#[test]
fn test_krum_via_aggregator_with_reputation() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);

    let updates = vec![
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[100.0]], // Byzantine
    ];
    let ids = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "attacker".to_string(),
    ];

    let _ = agg.aggregate(&updates, Some(&ids)).unwrap();

    // Krum selects one honest vector. The selected vector has distance 0
    // from the result, so it gets a reward. The attacker is far away -> penalized.
    assert!(
        agg.get_reputation("attacker") < agg.get_reputation("a"),
        "Attacker should have lower reputation"
    );
}

#[test]
fn test_krum_determinism_cross_invocation() {
    // Determinism: same inputs -> same output across separate aggregator instances
    let updates = vec![
        array![[1.0, 2.0, 3.0]],
        array![[1.1, 2.1, 3.1]],
        array![[0.9, 1.9, 2.9]],
        array![[1.05, 2.05, 3.05]],
        array![[50.0, 50.0, 50.0]],
    ];

    let mut results = Vec::new();
    for _ in 0..5 {
        let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);
        results.push(agg.aggregate(&updates, None).unwrap());
    }

    for r in &results[1..] {
        assert_eq!(
            &results[0], r,
            "Krum must produce identical results across invocations"
        );
    }
}

#[test]
fn test_krum_multidimensional_via_aggregator() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);

    // 3x2 updates
    let honest = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let updates = vec![
        honest.clone(),
        array![[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]],
        array![[0.9, 1.9], [2.9, 3.9], [4.9, 5.9]],
        array![[1.05, 2.05], [3.05, 4.05], [5.05, 6.05]],
        array![[99.0, 99.0], [99.0, 99.0], [99.0, 99.0]], // Byzantine
    ];

    let result = agg.aggregate(&updates, None).unwrap();
    assert_eq!(result.dim(), (3, 2), "Shape should be preserved");
    assert!(
        result[[0, 0]] < 2.0,
        "Should select honest vector, got {}",
        result[[0, 0]]
    );
}

#[test]
fn test_reputation_decay_toward_default() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);

    // Setup: penalize a client to low reputation
    let updates = vec![
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[100.0]], // Byzantine
    ];
    let ids = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "attacker".to_string(),
    ];
    let _ = agg.aggregate(&updates, Some(&ids)).unwrap();

    let before_decay = agg.get_reputation("attacker");
    assert!(before_decay < 0.5, "Attacker should be penalized");

    // Apply decay toward 0.5
    for _ in 0..10 {
        agg.decay_reputations(0.1);
    }

    let after_decay = agg.get_reputation("attacker");
    assert!(
        after_decay > before_decay,
        "Decay should move penalized client toward 0.5: {} -> {}",
        before_decay,
        after_decay
    );
    assert!(
        (after_decay - 0.5).abs() < 0.1,
        "After significant decay, score should be near 0.5, got {}",
        after_decay
    );
}

#[test]
fn test_reputation_decay_high_score() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);

    // Boost a client to high reputation
    let updates = vec![
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[100.0]],
    ];
    let ids = vec![
        "good".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "e".to_string(),
    ];
    // Run multiple rounds to boost "good" client reputation
    for _ in 0..10 {
        let _ = agg.aggregate(&updates, Some(&ids)).unwrap();
    }

    let before = agg.get_reputation("good");
    assert!(before > 0.5, "Good client should have high rep: {}", before);

    for _ in 0..20 {
        agg.decay_reputations(0.1);
    }

    let after = agg.get_reputation("good");
    assert!(
        after < before,
        "Decay should move high-rep client toward 0.5: {} -> {}",
        before,
        after
    );
}

#[test]
fn test_ban_gating_excludes_bad_clients() {
    // Use trimmed_mean to build accurate reputations (robust against attacker).
    // Then verify that ban gating actually excludes low-reputation clients.
    let mut agg = ByzantineAggregator::with_ban_threshold(AggregationMethod::TrimmedMean, 0.2, 0.3);

    let updates = vec![
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[100.0]], // Byzantine
    ];
    let ids = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "attacker".to_string(),
    ];

    // Run several rounds to drive attacker reputation below ban threshold.
    // Trimmed mean gives ~1.0, so attacker (100.0) has distance ~99 → penalized each round.
    for _ in 0..5 {
        let _ = agg.aggregate(&updates, Some(&ids)).unwrap();
    }

    assert!(
        agg.get_reputation("attacker") < 0.3,
        "Attacker should be below ban threshold: {}",
        agg.get_reputation("attacker")
    );
    assert!(
        agg.get_reputation("a") >= 0.3,
        "Honest client should be above ban threshold: {}",
        agg.get_reputation("a")
    );

    // Now the attacker is banned -- only honest clients participate
    let result = agg.aggregate(&updates, Some(&ids)).unwrap();
    assert!(
        (result[[0, 0]] - 1.0).abs() < 0.01,
        "With attacker banned, aggregation of honest clients should be ~1.0, got {}",
        result[[0, 0]]
    );
}

#[test]
fn test_ban_gating_fail_open() {
    // If ALL clients would be banned, should use all (fail-open)
    let mut agg = ByzantineAggregator::with_ban_threshold(AggregationMethod::FedAvg, 0.0, 0.99);

    // No one has reputation >= 0.99, so ban gating would exclude everyone
    let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];
    let ids = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    let result = agg.aggregate(&updates, Some(&ids)).unwrap();
    // Fail-open: all clients used
    assert!(
        (result[[0, 0]] - 2.0).abs() < 1e-6,
        "Fail-open should use all clients"
    );
}

#[test]
fn test_reputation_tracker_decay() {
    use qora_fl::reputation::ReputationTracker;

    let mut tracker = ReputationTracker::new();
    let peer = {
        let mut p = [0u8; 32];
        p[0] = 1;
        p
    };

    // Penalize to 0.1
    for _ in 0..5 {
        tracker.penalize_drift(&peer);
    }
    // 0.5 - 5*0.08 = 0.10
    assert!((tracker.get_score(&peer) - 0.10).abs() < 0.01);

    // Decay back toward 0.5
    for _ in 0..50 {
        tracker.decay_toward_default(0.1);
    }
    assert!(
        (tracker.get_score(&peer) - 0.5).abs() < 0.01,
        "After sufficient decay, should be near 0.5, got {}",
        tracker.get_score(&peer)
    );
}

#[test]
fn test_reputation_tracker_prune() {
    use qora_fl::reputation::ReputationTracker;

    let mut tracker = ReputationTracker::new();
    let peer1 = {
        let mut p = [0u8; 32];
        p[0] = 1;
        p
    };
    let peer2 = {
        let mut p = [0u8; 32];
        p[0] = 2;
        p
    };

    // peer1 gets rewarded (deviates from default)
    tracker.reward_valid_zkp(&peer1);
    // peer2 has no activity but gets a tiny bump then decays back
    tracker.reward_valid_zkp(&peer2);
    for _ in 0..50 {
        tracker.decay_toward_default(0.1);
    }

    // peer2 should be near default, peer1 also near default after decay
    tracker.prune_default_peers(0.01);
    assert_eq!(
        tracker.peer_count(),
        0,
        "Both peers should be pruned after heavy decay"
    );
}

// ===== Phase 3.5: BFP-16 Krum integration tests =====

#[test]
fn test_krum_returns_original_f32_values() {
    // Krum now returns the exact original f32 vector (no I16F16 quantization)
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);

    let updates = vec![
        array![[1.0, 2.0]],
        array![[1.1, 2.1]],
        array![[0.9, 1.9]],
        array![[1.05, 2.05]],
        array![[50.0, 50.0]],
    ];

    let result = agg.aggregate(&updates, None).unwrap();

    // Result must be exactly one of the honest input vectors (no quantization)
    let is_exact_match = updates[..4].iter().any(|u| u == &result);
    assert!(
        is_exact_match,
        "Result should be an exact copy of one input vector, got {:?}",
        result
    );
}

#[test]
fn test_krum_handles_large_weights() {
    // Values that exceed I16F16 range -- would have been clamped before BFP-16
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);

    let updates = vec![
        array![[50000.0, -40000.0]],
        array![[50001.0, -39999.0]],
        array![[49999.0, -40001.0]],
        array![[50000.5, -40000.5]],
        array![[999999.0, -999999.0]], // Byzantine
    ];

    let result = agg.aggregate(&updates, None).unwrap();
    assert!(
        result[[0, 0]] > 40000.0 && result[[0, 0]] < 60000.0,
        "Should select honest vector, got {}",
        result[[0, 0]]
    );
}

#[test]
fn test_krum_bfp16_parallel_determinism() {
    // Stress-test rayon determinism across many invocations
    let updates = vec![
        array![[1.0, 2.0, 3.0]],
        array![[1.1, 2.1, 3.1]],
        array![[0.9, 1.9, 2.9]],
        array![[1.05, 2.05, 3.05]],
        array![[50.0, 50.0, 50.0]],
    ];

    let mut results = Vec::new();
    for _ in 0..20 {
        let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);
        results.push(agg.aggregate(&updates, None).unwrap());
    }

    for r in &results[1..] {
        assert_eq!(
            &results[0], r,
            "Parallel BFP-16 Krum must be deterministic across invocations"
        );
    }
}

#[test]
fn test_krum_serde_roundtrip() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(2), 0.0);

    let updates = vec![
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[100.0]],
    ];
    let ids: Vec<String> = (0..7).map(|i| format!("c{}", i)).collect();
    let _ = agg.aggregate(&updates, Some(&ids)).unwrap();

    let json = serde_json::to_string(&agg).expect("serialize");
    let restored: ByzantineAggregator = serde_json::from_str(&json).expect("deserialize");

    assert!(
        (restored.get_reputation("c0") - agg.get_reputation("c0")).abs() < 1e-6,
        "Reputation should survive round-trip"
    );
}

// ===== Phase 5B: Multi-Krum integration tests =====

#[test]
fn test_multi_krum_via_aggregator() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, 3), 0.0);

    let updates = vec![
        array![[1.0, 2.0]],
        array![[1.1, 2.1]],
        array![[0.9, 1.9]],
        array![[1.05, 2.05]],
        array![[50.0, 50.0]], // Byzantine
    ];

    let result = agg.aggregate(&updates, None).unwrap();
    // Multi-Krum averages top-3, result should be close to honest center
    assert!(
        result[[0, 0]] < 2.0,
        "Multi-Krum should select honest vectors, got {}",
        result[[0, 0]]
    );
    assert!(
        result[[0, 1]] < 3.0,
        "Multi-Krum param 1 should be near honest, got {}",
        result[[0, 1]]
    );
}

#[test]
fn test_multi_krum_result_is_average() {
    // Multi-Krum should return an average, not an exact input vector
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(0, 3), 0.0);

    let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];

    let result = agg.aggregate(&updates, None).unwrap();
    // All 3 selected, average = (1+2+3)/3 = 2.0
    assert!(
        (result[[0, 0]] - 2.0).abs() < 0.01,
        "Multi-Krum m=n should average all, got {}",
        result[[0, 0]]
    );
}

#[test]
fn test_multi_krum_too_few_clients() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, 3), 0.0);

    let updates = vec![array![[1.0]], array![[2.0]]];
    let result = agg.aggregate(&updates, None);
    assert!(result.is_err(), "Multi-Krum with <3 clients should error");
}

#[test]
fn test_multi_krum_serde_roundtrip() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, 3), 0.0);

    let updates = vec![
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[100.0]],
    ];
    let ids: Vec<String> = (0..5).map(|i| format!("c{}", i)).collect();
    let _ = agg.aggregate(&updates, Some(&ids)).unwrap();

    let json = serde_json::to_string(&agg).expect("serialize");
    let restored: ByzantineAggregator = serde_json::from_str(&json).expect("deserialize");

    assert!(
        (restored.get_reputation("c0") - agg.get_reputation("c0")).abs() < 1e-6,
        "Multi-Krum reputation should survive round-trip"
    );
}

#[test]
fn test_multi_krum_attack_resistance() {
    // 4 honest + 1 Byzantine, Multi-Krum m=3 should pick 3 honest
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, 3), 0.0);

    let updates = vec![
        array![[1.0, 1.0]],
        array![[1.1, 0.9]],
        array![[0.9, 1.1]],
        array![[1.05, 0.95]],
        array![[100.0, 100.0]], // Byzantine
    ];
    let ids = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "attacker".to_string(),
    ];

    let result = agg.aggregate(&updates, Some(&ids)).unwrap();
    assert!(
        result[[0, 0]] < 2.0,
        "Multi-Krum should resist attack, got {}",
        result[[0, 0]]
    );
    assert!(
        agg.get_reputation("attacker") < agg.get_reputation("a"),
        "Attacker reputation should be lower"
    );
}

// ===== BFP-16 Accuracy Analysis =====

#[test]
fn test_bfp16_krum_index_agreement() {
    use qora_fl::aggregators::krum::{aggregate_krum_bfp16, Bfp16Vec};
    use rand::prelude::*;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let n_trials = 100;
    let n_clients = 7;
    let dim = 100;
    let f = 1;
    let k = n_clients - f - 2; // 4 nearest neighbors
    let mut agreements = 0;

    for _ in 0..n_trials {
        // Generate random f32 vectors (uniform [-1, 1])
        let vectors: Vec<Vec<f32>> = (0..n_clients)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();

        // f64-precision brute-force Krum
        let f64_best = (0..n_clients)
            .min_by(|&i, &j| {
                let score_i: f64 = {
                    let mut dists: Vec<f64> = (0..n_clients)
                        .filter(|&x| x != i)
                        .map(|x| {
                            vectors[i]
                                .iter()
                                .zip(&vectors[x])
                                .map(|(a, b)| {
                                    let diff = (*a as f64) - (*b as f64);
                                    diff * diff
                                })
                                .sum()
                        })
                        .collect();
                    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    dists[..k].iter().sum()
                };
                let score_j: f64 = {
                    let mut dists: Vec<f64> = (0..n_clients)
                        .filter(|&x| x != j)
                        .map(|x| {
                            vectors[j]
                                .iter()
                                .zip(&vectors[x])
                                .map(|(a, b)| {
                                    let diff = (*a as f64) - (*b as f64);
                                    diff * diff
                                })
                                .sum()
                        })
                        .collect();
                    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    dists[..k].iter().sum()
                };
                score_i.partial_cmp(&score_j).unwrap()
            })
            .unwrap();

        // BFP-16 Krum
        let bfp_vecs: Vec<Bfp16Vec> = vectors
            .iter()
            .map(|v| Bfp16Vec::from_f32_slice(v))
            .collect();
        let bfp16_best = aggregate_krum_bfp16(&bfp_vecs, f).unwrap();

        if f64_best == bfp16_best {
            agreements += 1;
        }
    }

    let agreement_rate = agreements as f64 / n_trials as f64;
    assert!(
        agreement_rate >= 0.90,
        "BFP-16 Krum agreement rate {:.1}% is below 90% threshold",
        agreement_rate * 100.0
    );
}

#[test]
fn test_bfp16_roundtrip_quantization_error() {
    use qora_fl::aggregators::krum::Bfp16Vec;
    use rand::prelude::*;

    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let n_trials = 200;
    let dim = 500;
    let mut total_rel_error = 0.0f64;
    let mut count = 0u64;

    for trial in 0..n_trials {
        // Vary scale: 1e-2 to 1e2
        let scale = 10.0f32.powf((trial as f32 / n_trials as f32) * 4.0 - 2.0);
        let original: Vec<f32> = (0..dim)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale)
            .collect();

        let bfp = Bfp16Vec::from_f32_slice(&original);
        let recovered = bfp.to_vec_f32();

        for (o, r) in original.iter().zip(recovered.iter()) {
            if o.abs() > scale * 0.01 {
                // Only measure relative error for values not near zero relative to block max
                let rel = ((*o - *r) as f64).abs() / (*o as f64).abs();
                total_rel_error += rel;
                count += 1;
            }
        }
    }

    let mean_rel_error = total_rel_error / count as f64;
    assert!(
        mean_rel_error < 0.01,
        "Mean relative error {:.6} exceeds 1% threshold",
        mean_rel_error
    );
}

#[test]
fn test_bfp16_krum_agreement_with_byzantine() {
    use qora_fl::aggregators::krum::{aggregate_krum_bfp16, Bfp16Vec};
    use rand::prelude::*;

    let mut rng = rand::rngs::StdRng::seed_from_u64(99);
    let n_trials = 100;
    let n_clients = 7;
    let dim = 100;
    let f = 1;
    let k = n_clients - f - 2;
    let mut agreements = 0;

    for _ in 0..n_trials {
        // 6 honest vectors near origin, 1 Byzantine at 100x
        let mut vectors: Vec<Vec<f32>> = (0..n_clients - 1)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();
        vectors.push((0..dim).map(|_| 100.0 + rng.gen::<f32>()).collect());

        // f64-precision brute-force Krum
        let f64_best = (0..n_clients)
            .min_by(|&i, &j| {
                let score = |idx: usize| -> f64 {
                    let mut dists: Vec<f64> = (0..n_clients)
                        .filter(|&x| x != idx)
                        .map(|x| {
                            vectors[idx]
                                .iter()
                                .zip(&vectors[x])
                                .map(|(a, b)| {
                                    let d = (*a as f64) - (*b as f64);
                                    d * d
                                })
                                .sum()
                        })
                        .collect();
                    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    dists[..k].iter().sum()
                };
                score(i).partial_cmp(&score(j)).unwrap()
            })
            .unwrap();

        // BFP-16 Krum
        let bfp_vecs: Vec<Bfp16Vec> = vectors
            .iter()
            .map(|v| Bfp16Vec::from_f32_slice(v))
            .collect();
        let bfp16_best = aggregate_krum_bfp16(&bfp_vecs, f).unwrap();

        if f64_best == bfp16_best {
            agreements += 1;
        }
    }

    let agreement_rate = agreements as f64 / n_trials as f64;
    assert!(
        agreement_rate >= 0.95,
        "BFP-16 Krum with Byzantine agreement rate {:.1}% is below 95% threshold",
        agreement_rate * 100.0
    );
}
