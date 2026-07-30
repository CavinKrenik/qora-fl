//! Regression tests for input validation hardening.
//!
//! Each test here pins down a failure that the library previously accepted
//! silently. The behavior being regressed against was measured before the fix
//! and is recorded in the comment above each test.

use ndarray::array;
use ndarray::Array2;
use qora_fl::aggregators::krum::{aggregate_krum, aggregate_krum_bfp16, Bfp16Vec};
use qora_fl::aggregators::{AggregationMethod, ByzantineAggregator};
use qora_fl::error::QoraError;
use qora_fl::verification::{krum_condition_met, krum_min_clients};
use qora_fl::{fedavg, median, trimmed_mean};

/// Five honest updates plus one carrying `bad` in every coordinate.
fn updates_with(bad: f32) -> Vec<Array2<f32>> {
    vec![
        array![[1.0, 1.0]],
        array![[1.1, 0.9]],
        array![[0.9, 1.1]],
        array![[1.05, 0.95]],
        array![[bad, bad]],
    ]
}

// ===== Non-finite updates: free functions =====

/// Before: the outcome depended on *which client* sent the NaN. With 5 clients
/// and trim 0.2, NaN at index 0 or 4 was trimmed away and the result was 1.0;
/// at index 1, 2, or 3 it survived into the mean and the result was NaN.
/// `partial_cmp(...).unwrap_or(Equal)` made NaN compare equal to everything, so
/// the stable sort left it wherever the caller happened to put it.
#[test]
fn trimmed_mean_rejects_nan_at_every_client_position() {
    for nan_pos in 0..5 {
        let mut updates: Vec<Array2<f32>> = (0..5).map(|_| array![[1.0f32]]).collect();
        updates[nan_pos] = array![[f32::NAN]];

        match trimmed_mean(&updates, 0.2) {
            Err(QoraError::NonFiniteValue {
                update_index,
                value,
                ..
            }) => {
                assert_eq!(
                    update_index, nan_pos,
                    "error should attribute the NaN to the client that sent it"
                );
                assert!(value.is_nan());
            }
            other => panic!("NaN at position {} was not rejected: {:?}", nan_pos, other),
        }
    }
}

#[test]
fn trimmed_mean_rejects_infinity() {
    for bad in [f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            matches!(
                trimmed_mean(&updates_with(bad), 0.2),
                Err(QoraError::NonFiniteValue {
                    update_index: 4,
                    ..
                })
            ),
            "trimmed_mean should reject {}",
            bad
        );
    }
}

/// Before: returned `Ok([1.05, 1.0])` -- a finite, plausible-looking result
/// silently computed from a corrupted coordinate sort.
#[test]
fn median_rejects_nan() {
    match median(&updates_with(f32::NAN)) {
        Err(QoraError::NonFiniteValue {
            update_index,
            value,
            ..
        }) => {
            assert_eq!(update_index, 4);
            assert!(value.is_nan());
        }
        other => panic!("median accepted NaN: {:?}", other),
    }
}

#[test]
fn median_rejects_infinity() {
    for bad in [f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            matches!(
                median(&updates_with(bad)),
                Err(QoraError::NonFiniteValue {
                    update_index: 4,
                    ..
                })
            ),
            "median should reject {}",
            bad
        );
    }
}

/// Before: returned `Ok([NaN, NaN])` -- one client destroyed every coordinate.
#[test]
fn fedavg_rejects_nan() {
    assert!(matches!(
        fedavg(&updates_with(f32::NAN), None),
        Err(QoraError::NonFiniteValue {
            update_index: 4,
            ..
        })
    ));
}

/// Before: returned `Ok([inf, inf])` / `Ok([-inf, -inf])`.
#[test]
fn fedavg_rejects_infinity() {
    for bad in [f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            matches!(
                fedavg(&updates_with(bad), None),
                Err(QoraError::NonFiniteValue {
                    update_index: 4,
                    ..
                })
            ),
            "fedavg should reject {}",
            bad
        );
    }
}

/// Updates can be entirely well-formed while a single non-finite *weight*
/// corrupts the weighted mean. Before: NaN weight -> NaN output.
#[test]
fn fedavg_rejects_non_finite_weight() {
    let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let weights = vec![1.0, bad, 1.0];
        match fedavg(&updates, Some(&weights)) {
            Err(QoraError::NonFiniteWeight { index, .. }) => assert_eq!(index, 1),
            other => panic!("fedavg accepted weight {}: {:?}", bad, other),
        }
    }

    // The finite-weight path still works.
    let ok = fedavg(&updates, Some(&[1.0, 1.0, 2.0])).unwrap();
    assert!((ok[[0, 0]] - 2.25).abs() < 1e-6);
}

// ===== Non-finite updates: high-level aggregator, every method =====

/// Before: TrimmedMean/Median/Krum/MultiKrum returned finite-looking results
/// and FedAvg returned NaN. None of them reported an error.
#[test]
fn aggregate_rejects_nan() {
    let methods = [
        AggregationMethod::TrimmedMean,
        AggregationMethod::Median,
        AggregationMethod::FedAvg,
        AggregationMethod::Krum(1),
        AggregationMethod::MultiKrum(1, 3),
    ];

    for method in methods {
        let mut agg = ByzantineAggregator::new(method.clone(), 0.2);
        assert!(
            matches!(
                agg.aggregate(&updates_with(f32::NAN), None),
                Err(QoraError::NonFiniteValue {
                    update_index: 4,
                    ..
                })
            ),
            "{:?} should reject NaN",
            method
        );
    }
}

#[test]
fn aggregate_rejects_infinity() {
    let methods = [
        AggregationMethod::TrimmedMean,
        AggregationMethod::Median,
        AggregationMethod::FedAvg,
        AggregationMethod::Krum(1),
        AggregationMethod::MultiKrum(1, 3),
    ];

    for method in methods {
        for bad in [f32::INFINITY, f32::NEG_INFINITY] {
            let mut agg = ByzantineAggregator::new(method.clone(), 0.2);
            assert!(
                matches!(
                    agg.aggregate(&updates_with(bad), None),
                    Err(QoraError::NonFiniteValue {
                        update_index: 4,
                        ..
                    })
                ),
                "{:?} should reject {}",
                method,
                bad
            );
        }
    }
}

/// The BFP-16 encoder quantizes NaN to a zero mantissa, so Krum's distance
/// metric could not see the poisoning at all. Before: `Krum` over five all-NaN
/// updates returned `Ok([[NaN]])`.
#[test]
fn aggregate_krum_rejects_all_nan_updates() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);
    let all_nan = vec![array![[f32::NAN]]; 5];
    assert!(matches!(
        agg.aggregate(&all_nan, None),
        Err(QoraError::NonFiniteValue {
            update_index: 0,
            ..
        })
    ));
}

/// Documents the laundering behavior that makes the check above necessary, and
/// pins it so a future change to `from_f32_slice` is a deliberate decision
/// rather than an accident. NaN encodes to mantissa 0, i.e. reads as 0.0.
#[test]
fn bfp16_encoding_still_launders_non_finite_input() {
    let nan_encoded = Bfp16Vec::from_f32_slice(&[1.0, f32::NAN, 2.0]);
    assert_eq!(
        nan_encoded.mantissas[1], 0,
        "NaN quantizes to a zero mantissa -- the reason validation must run \
         before encoding, not inside it"
    );

    let inf_encoded = Bfp16Vec::from_f32_slice(&[1.0, f32::INFINITY, 2.0]);
    assert_eq!(
        inf_encoded.exponent, 126,
        "infinity saturates the shared exponent"
    );
    assert_eq!(
        inf_encoded.mantissas[0], 0,
        "a saturated exponent zeroes every other coordinate in the vector"
    );
}

// ===== Client ID / update count agreement =====

/// Before: panicked with "index out of bounds: the len is 3 but the index is 3"
/// inside the ban-threshold filter -- a panic across the PyO3 boundary too.
#[test]
fn aggregator_rejects_too_many_client_ids() {
    let mut agg = ByzantineAggregator::with_ban_threshold(AggregationMethod::Median, 0.0, 0.2);
    let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];
    let ids: Vec<String> = (0..6).map(|i| format!("c{}", i)).collect();

    assert!(matches!(
        agg.aggregate(&updates, Some(&ids)),
        Err(QoraError::ClientIdCountMismatch {
            updates: 3,
            client_ids: 6
        })
    ));
}

/// Before: silently succeeded, aggregating all updates but attributing
/// reputation to only the first `client_ids.len()` of them -- the `zip` in
/// `update_reputations` stops at the shorter side, so unnamed clients were
/// scored by nobody.
#[test]
fn aggregator_rejects_too_few_client_ids() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0);
    let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];
    let ids = vec!["c0".to_string()];

    assert!(matches!(
        agg.aggregate(&updates, Some(&ids)),
        Err(QoraError::ClientIdCountMismatch {
            updates: 3,
            client_ids: 1
        })
    ));
}

#[test]
fn aggregate_rejects_mismatched_client_ids() {
    // Mismatch is rejected regardless of reputation gating being enabled.
    let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];

    for ban_threshold in [0.0, 0.2] {
        let mut agg =
            ByzantineAggregator::with_ban_threshold(AggregationMethod::Median, 0.0, ban_threshold);
        let ids = vec!["a".to_string(), "b".to_string()];
        assert!(
            agg.aggregate(&updates, Some(&ids)).is_err(),
            "mismatch should be rejected at ban_threshold={}",
            ban_threshold
        );
    }

    // Matching lengths still work.
    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0);
    let ids: Vec<String> = (0..3).map(|i| format!("c{}", i)).collect();
    assert!(agg.aggregate(&updates, Some(&ids)).is_ok());
}

// ===== Krum quorum condition =====

/// Before: printed "WARN: Krum condition not met ... Proceeding with
/// best-effort." to stderr and returned a result carrying no Byzantine
/// guarantee, indistinguishable from a sound one.
#[test]
fn aggregator_krum_rejects_invalid_condition() {
    // n=4, f=2 -> needs 2*2+3 = 7
    let updates = vec![array![[1.0]], array![[1.1]], array![[0.9]], array![[1.05]]];

    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(2), 0.0);
    match agg.aggregate(&updates, None) {
        Err(QoraError::InsufficientQuorum { needed, actual }) => {
            assert_eq!(needed, 7, "needed must be 2f+3, not a hardcoded 3");
            assert_eq!(actual, 4);
        }
        other => panic!("Krum(2) with n=4 should be refused: {:?}", other),
    }

    // n=5, f=1 is exactly at the boundary and must still succeed.
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0);
    let boundary = vec![
        array![[1.0]],
        array![[1.1]],
        array![[0.9]],
        array![[1.05]],
        array![[100.0]],
    ];
    assert!(
        agg.aggregate(&boundary, None).is_ok(),
        "n = 2f+3 exactly should be accepted"
    );
}

#[test]
fn aggregator_multi_krum_rejects_invalid_condition() {
    // n=6, f=2 -> needs 7
    let updates: Vec<Array2<f32>> = (0..6).map(|i| array![[1.0 + i as f32 * 0.1]]).collect();

    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(2, 3), 0.0);
    match agg.aggregate(&updates, None) {
        Err(QoraError::InsufficientQuorum { needed, actual }) => {
            assert_eq!(needed, 7);
            assert_eq!(actual, 6);
        }
        other => panic!("MultiKrum(2, 3) with n=6 should be refused: {:?}", other),
    }

    // n=7, f=2 is at the boundary and must succeed.
    let updates: Vec<Array2<f32>> = (0..7).map(|i| array![[1.0 + i as f32 * 0.1]]).collect();
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(2, 3), 0.0);
    assert!(agg.aggregate(&updates, None).is_ok());
}

/// The low-level APIs keep their `Option` return; `None` now means exactly
/// "Krum's condition is not satisfiable", with no best-effort fallthrough.
#[test]
fn low_level_krum_returns_none_when_condition_is_invalid() {
    use fixed::types::I16F16;

    // I16F16 path: n=4, f=2 -> needs 7.
    let vectors: Vec<Vec<I16F16>> = [1.0, 1.1, 100.0, 100.0]
        .iter()
        .map(|&v| vec![I16F16::from_num(v)])
        .collect();
    assert!(
        aggregate_krum(&vectors, 2).is_none(),
        "n=4, f=2 must be refused, not answered best-effort"
    );
    assert!(
        aggregate_krum(&vectors, 0).is_some(),
        "n=4, f=0 satisfies n >= 3 and must still work"
    );

    // BFP-16 path: same condition.
    let bfp: Vec<Bfp16Vec> = [1.0f32, 1.1, 100.0, 100.0]
        .iter()
        .map(|&v| Bfp16Vec::from_f32_slice(&[v]))
        .collect();
    assert!(aggregate_krum_bfp16(&bfp, 2).is_none());
    assert!(aggregate_krum_bfp16(&bfp, 0).is_some());

    // Both paths agree with the documented predicate across a grid.
    for n in 0..12usize {
        for f in 0..5usize {
            let vs: Vec<Vec<I16F16>> = (0..n).map(|i| vec![I16F16::from_num(i as f32)]).collect();
            let bs: Vec<Bfp16Vec> = (0..n)
                .map(|i| Bfp16Vec::from_f32_slice(&[i as f32]))
                .collect();
            let expected = krum_condition_met(n, f);
            assert_eq!(
                aggregate_krum(&vs, f).is_some(),
                expected,
                "aggregate_krum disagrees with krum_condition_met at n={}, f={}",
                n,
                f
            );
            assert_eq!(
                aggregate_krum_bfp16(&bs, f).is_some(),
                expected,
                "aggregate_krum_bfp16 disagrees with krum_condition_met at n={}, f={}",
                n,
                f
            );
        }
    }
}

/// An absurd `f` can reach the aggregator from untrusted input (e.g. a parsed
/// `"krum:18446744073709551615"` method string). `2 * f + 3` must saturate
/// rather than overflow.
#[test]
fn krum_min_clients_saturates_on_absurd_f() {
    assert_eq!(krum_min_clients(usize::MAX), usize::MAX);

    let updates = vec![array![[1.0]]; 5];
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(usize::MAX), 0.0);
    assert!(agg.aggregate(&updates, None).is_err());
}

// ===== Reputation tracking =====

/// Before: the NaN client's score stayed at the 0.5 default while honest
/// clients rose to 0.52. `distance` was NaN, and NaN satisfies neither
/// `distance < 1.0` nor `distance > 10.0`, so a NaN attacker was never
/// penalized -- and therefore never bannable. Rejecting the round is what
/// closes that hole.
#[test]
fn nan_update_does_not_enter_reputation_tracking() {
    let mut agg = ByzantineAggregator::with_ban_threshold(AggregationMethod::TrimmedMean, 0.2, 0.2);

    let updates = vec![
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[1.0]],
        array![[f32::NAN]],
    ];
    let ids: Vec<String> = (0..5).map(|i| format!("c{}", i)).collect();

    assert!(agg.aggregate(&updates, Some(&ids)).is_err());

    // No score moved: the round was refused before reputation was touched, so
    // the NaN client gained no standing and the honest clients lost none.
    for i in 0..5 {
        assert_eq!(
            agg.get_reputation(&format!("c{}", i)),
            0.5,
            "c{} reputation must be untouched by a rejected round",
            i
        );
    }
}

/// A rejected round must not leave the aggregator in a state where a later
/// valid round behaves differently.
#[test]
fn rejected_round_does_not_corrupt_later_rounds() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);
    let ids: Vec<String> = (0..5).map(|i| format!("c{}", i)).collect();

    let mut bad = vec![array![[1.0f32]]; 5];
    bad[2] = array![[f32::NAN]];
    assert!(agg.aggregate(&bad, Some(&ids)).is_err());

    let good = vec![array![[1.0f32]]; 5];
    let result = agg.aggregate(&good, Some(&ids)).unwrap();
    assert!((result[[0, 0]] - 1.0).abs() < 1e-6);
    assert!(
        agg.get_reputation("c2") > 0.5,
        "c2 should be rewarded on the valid round"
    );
}
