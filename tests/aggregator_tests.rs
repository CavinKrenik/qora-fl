//! Integration tests for Qora-FL aggregation algorithms

use ndarray::array;
use ndarray::Array2;
use qora_fl::aggregators::{AggregationMethod, ByzantineAggregator};
use qora_fl::{fedavg, median, trimmed_mean};

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
