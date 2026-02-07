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

    // All should be close to [1.0, 2.0]
    for result in &[&tm, &med, &avg] {
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
