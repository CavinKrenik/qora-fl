//! Optional norm-bound filtering in the aggregation path.
//!
//! The governing rule these tests pin:
//!
//! > With no norm bound configured, aggregation behaves exactly as it did
//! > before the filter existed. With a valid bound configured, over-bound
//! > updates are excluded, every exclusion is auditable, and an empty accepted
//! > cohort fails closed.
//!
//! The audit record itself is covered in `tests/audited_aggregation.rs`; this
//! file covers configuration, the filtering decision, and its consequences for
//! weights, reputation, and method preconditions.

use ndarray::{array, Array2};
use qora_fl::aggregators::{AggregationMethod, ByzantineAggregator};
use qora_fl::{fedavg, median, QoraError};

/// Build an aggregator with exact reputation scores and an optional bound.
///
/// `ByzantineAggregator` exposes no score setter, and driving scores through
/// repeated rounds would couple these tests to the reputation arithmetic they
/// are meant to leave alone. Deserialization sets the state directly.
fn aggregator_with_scores(
    method: &str,
    ban_threshold: f32,
    scores: &[(&str, f32)],
    norm_bound: Option<f32>,
) -> ByzantineAggregator {
    let entries: Vec<String> = scores
        .iter()
        .map(|(id, score)| format!(r#""{}":{}"#, id, score))
        .collect();
    let bound = norm_bound.map_or("null".to_string(), |b| b.to_string());
    let json = format!(
        r#"{{"method":{},"trim_fraction":0.0,"reputation":{{{}}},"ban_threshold":{},"adaptive_trim":false,"norm_bound":{}}}"#,
        method,
        entries.join(","),
        ban_threshold,
        bound
    );
    serde_json::from_str(&json).expect("aggregator fixture must deserialize")
}

fn ids(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

/// Five well-behaved updates plus one whose norm is enormous.
fn cohort_with_one_outlier() -> Vec<Array2<f32>> {
    vec![
        array![[1.0, 1.0]],
        array![[1.0, 2.0]],
        array![[2.0, 1.0]],
        array![[2.0, 2.0]],
        array![[1.5, 1.5]],
        array![[900.0, 900.0]], // norm ~1272.8
    ]
}

// ===== Disabled by default =====

#[test]
fn filtering_is_off_for_every_constructor() {
    assert_eq!(
        ByzantineAggregator::new(AggregationMethod::Median, 0.0).norm_bound(),
        None
    );
    assert_eq!(
        ByzantineAggregator::with_ban_threshold(AggregationMethod::Median, 0.0, 0.2)
            .unwrap()
            .norm_bound(),
        None
    );
}

#[test]
fn a_disabled_aggregator_matches_the_unfiltered_functions() {
    // The free functions have no filtering at all, so agreeing with them is
    // the strongest available statement that nothing was excluded.
    let updates = cohort_with_one_outlier();

    let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0);
    assert_eq!(
        agg.aggregate(&updates, None).unwrap(),
        fedavg(&updates, None).unwrap()
    );

    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0);
    assert_eq!(
        agg.aggregate(&updates, None).unwrap(),
        median(&updates).unwrap()
    );
}

#[test]
fn a_disabled_aggregator_includes_an_update_a_bound_would_reject() {
    let updates = cohort_with_one_outlier();

    let mut unfiltered = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0);
    let mut filtered = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();

    let with_outlier = unfiltered.aggregate(&updates, None).unwrap();
    let without_outlier = filtered.aggregate(&updates, None).unwrap();

    assert!(
        with_outlier[[0, 0]] > 100.0,
        "the outlier must still dominate an unfiltered mean, got {}",
        with_outlier[[0, 0]]
    );
    assert!(
        without_outlier[[0, 0]] < 3.0,
        "the filtered mean must exclude it, got {}",
        without_outlier[[0, 0]]
    );
}

#[test]
fn a_disabled_aggregator_reports_the_same_errors() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0);

    assert!(matches!(
        agg.aggregate(&[], None),
        Err(QoraError::EmptyUpdates)
    ));
    assert!(matches!(
        agg.aggregate(&[array![[1.0, 2.0]], array![[1.0]]], None),
        Err(QoraError::DimensionMismatch)
    ));
    assert!(matches!(
        agg.aggregate(&[array![[1.0]], array![[f32::NAN]]], None),
        Err(QoraError::NonFiniteValue {
            update_index: 1,
            ..
        })
    ));
    assert!(matches!(
        agg.aggregate(&[array![[1.0]]], Some(&ids(&["a", "b"]))),
        Err(QoraError::ClientIdCountMismatch {
            updates: 1,
            client_ids: 2
        })
    ));
}

// ===== Configuration =====

#[test]
fn a_valid_bound_is_accepted_and_readable() {
    for bound in [f32::MIN_POSITIVE, 1e-20, 0.5, 1.0, 1e20, f32::MAX] {
        let agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0)
            .with_norm_bound_filter(bound)
            .unwrap_or_else(|e| panic!("bound {} must be accepted: {}", bound, e));
        assert_eq!(agg.norm_bound(), Some(bound));
    }
}

#[test]
fn an_unusable_bound_is_refused_rather_than_clamped() {
    for bad in [0.0, -1.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        match ByzantineAggregator::new(AggregationMethod::Median, 0.0).with_norm_bound_filter(bad) {
            Err(QoraError::InvalidNormBound { value }) => {
                assert_eq!(value.is_nan(), bad.is_nan(), "bound {}", bad);
            }
            Ok(agg) => panic!(
                "bound {} must be refused, got a filter configured at {:?}",
                bad,
                agg.norm_bound()
            ),
            Err(other) => panic!("bound {} gave the wrong error: {:?}", bad, other),
        }
    }
}

#[test]
fn disabling_restores_the_unfiltered_result() {
    let updates = cohort_with_one_outlier();

    let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap()
        .without_norm_bound_filter();

    assert_eq!(agg.norm_bound(), None);
    assert_eq!(
        agg.aggregate(&updates, None).unwrap(),
        fedavg(&updates, None).unwrap()
    );
}

// ===== Serialized compatibility =====

#[test]
fn a_configuration_without_the_field_restores_disabled() {
    // Exactly what 0.3.1 wrote, before the field existed.
    let json = r#"{"method":"FedAvg","trim_fraction":0.0,"reputation":{"a":0.5},"ban_threshold":0.0,"adaptive_trim":false}"#;
    let mut agg: ByzantineAggregator =
        serde_json::from_str(json).expect("a 0.3.1 configuration must still restore");

    assert_eq!(agg.norm_bound(), None);

    let updates = cohort_with_one_outlier();
    assert_eq!(
        agg.aggregate(&updates, None).unwrap(),
        fedavg(&updates, None).unwrap(),
        "restoring an old configuration must not enable filtering"
    );
}

#[test]
fn an_explicit_null_restores_disabled() {
    let agg = aggregator_with_scores(r#""FedAvg""#, 0.0, &[("a", 0.5)], None);
    assert_eq!(agg.norm_bound(), None);
}

#[test]
fn a_valid_persisted_bound_restores_enabled() {
    let agg = aggregator_with_scores(r#""FedAvg""#, 0.0, &[("a", 0.5)], Some(12.5));
    assert_eq!(agg.norm_bound(), Some(12.5));
}

#[test]
fn a_persisted_bound_faces_the_same_validation_as_a_constructed_one() {
    // Restoring is a configuration path like any other. A stored 0.0 would
    // otherwise reject every update at the next round, silently.
    for bad in ["0.0", "-1.0", "-0.5"] {
        let json = format!(
            r#"{{"method":"FedAvg","trim_fraction":0.0,"reputation":{{}},"ban_threshold":0.0,"adaptive_trim":false,"norm_bound":{}}}"#,
            bad
        );
        assert!(
            serde_json::from_str::<ByzantineAggregator>(&json).is_err(),
            "persisted bound {} must be rejected",
            bad
        );
    }
}

#[test]
fn an_out_of_range_persisted_magnitude_does_not_restore() {
    // `1e400` is beyond f64 range; whether serde rejects the number outright or
    // widens it to infinity, it must not produce a usable filter.
    let json = r#"{"method":"FedAvg","trim_fraction":0.0,"reputation":{},"ban_threshold":0.0,"adaptive_trim":false,"norm_bound":1e400}"#;
    assert!(serde_json::from_str::<ByzantineAggregator>(json).is_err());
}

#[test]
fn round_trip_preserves_both_states() {
    for bound in [None, Some(7.5)] {
        let original = match bound {
            None => ByzantineAggregator::new(AggregationMethod::Median, 0.0),
            Some(b) => ByzantineAggregator::new(AggregationMethod::Median, 0.0)
                .with_norm_bound_filter(b)
                .unwrap(),
        };

        let json = serde_json::to_string(&original).unwrap();
        let restored: ByzantineAggregator = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.norm_bound(), bound, "round-tripped {:?}", bound);
    }
}

// ===== Boundary behavior =====

#[test]
fn below_at_and_above_the_bound() {
    // Norms 3.0, 5.0 and 5.0000005 against a bound of exactly 5.0. Equality is
    // acceptance, matching `check_norm_bound`; a disagreement between the public
    // checker and the integrated filter would be invisible everywhere else.
    let updates = vec![
        array![[3.0, 0.0]], // norm 3.0   -- below
        array![[3.0, 4.0]], // norm 5.0   -- exactly at
        array![[0.0, 6.0]], // norm 6.0   -- above
    ];

    let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(5.0)
        .unwrap();

    let result = agg.aggregate(&updates, None).unwrap();
    // Mean of the two accepted updates: [(3+3)/2, (0+4)/2] = [3.0, 2.0].
    assert!(
        (result[[0, 0]] - 3.0).abs() < 1e-6,
        "got {}",
        result[[0, 0]]
    );
    assert!(
        (result[[0, 1]] - 2.0).abs() < 1e-6,
        "got {}",
        result[[0, 1]]
    );
}

#[test]
fn very_large_finite_norms_are_compared_exactly() {
    // True norm 1e20 * sqrt(2) ~= 1.41e20: four orders of magnitude below the
    // 1e30 bound. An f32 squared sum would overflow and reject it.
    let updates = vec![array![[1e20f32, 1e20f32]], array![[1.0, 1.0]]];

    let mut permissive = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(1e30)
        .unwrap();
    let result = permissive.aggregate(&updates, None).unwrap();
    assert!(
        result[[0, 0]] > 1e19,
        "a 1e20 update inside a 1e30 bound must participate, got {}",
        result[[0, 0]]
    );

    let mut strict = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(1e19)
        .unwrap();
    let result = strict.aggregate(&updates, None).unwrap();
    assert!(
        (result[[0, 0]] - 1.0).abs() < 1e-6,
        "the same update must be excluded by a bound it genuinely exceeds, got {}",
        result[[0, 0]]
    );
}

#[test]
fn very_small_finite_norms_stay_positive() {
    // True norm ~1.41e-25. An f32 squared sum flushes to zero, which passes
    // every positive bound -- a false accept.
    let updates = vec![array![[1e-25f32, 1e-25f32]], array![[1e-30f32, 0.0]]];

    let mut strict = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(1e-26)
        .unwrap();
    let result = strict.aggregate(&updates, None).unwrap();
    assert_eq!(
        result[[0, 0]],
        1e-30,
        "the 1e-25 update exceeds a 1e-26 bound and must be excluded"
    );

    let mut permissive = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(1e-24)
        .unwrap();
    assert!(permissive.aggregate(&updates, None).unwrap()[[0, 0]] > 1e-26);
}

#[test]
fn non_finite_coordinates_are_a_validation_error_not_a_rejection() {
    // A poisoned update must not be quietly filtered out as "oversized": the
    // caller needs to hear that the batch was malformed.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
            .with_norm_bound_filter(1.0)
            .unwrap();

        let updates = vec![array![[0.5]], array![[bad]], array![[0.5]]];
        assert!(
            matches!(
                agg.aggregate(&updates, None),
                Err(QoraError::NonFiniteValue {
                    update_index: 1,
                    ..
                })
            ),
            "{} must be reported as malformed input",
            bad
        );
    }
}

#[test]
fn an_empty_batch_outranks_the_filter() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(1.0)
        .unwrap();
    assert!(matches!(
        agg.aggregate(&[], None),
        Err(QoraError::EmptyUpdates)
    ));
}

// ===== Alignment =====

#[test]
fn fedavg_weights_leave_with_their_clients() {
    // A: update 0, weight 1, accepted.
    // B: huge update, weight 100, rejected.
    // C: update 10, weight 9, accepted.
    //
    // Expected (0*1 + 10*9) / (1 + 9) = 9.0. A shifted weight vector would give
    // C the rejected weight of 100, and leaving 100 in the denominator would
    // give 0.9.
    let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(100.0)
        .unwrap();

    let updates = vec![array![[0.0]], array![[1000.0]], array![[10.0]]];
    let result = agg
        .aggregate_weighted(
            &updates,
            Some(&ids(&["a", "b", "c"])),
            Some(&[1.0, 100.0, 9.0]),
        )
        .unwrap();

    assert!(
        (result[[0, 0]] - 9.0).abs() < 1e-4,
        "expected 9.0, got {} -- the rejected weight must leave the numerator \
         and the denominator with its client",
        result[[0, 0]]
    );
}

#[test]
fn client_ids_stay_aligned_after_a_rejection() {
    // "b" is rejected by the bound, leaving [a, c, d, e, f] against updates
    // [0, 0, 60, 0, 0]. Every survivor but "d" agrees with the median and gains;
    // "d" deviates and loses. A shifted ID vector would score [a, b, c, d, e]
    // against those updates instead, moving "d"'s penalty onto "c" and scoring
    // a client that never participated.
    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0)
        .with_norm_bound_filter(100.0)
        .unwrap();

    let updates = vec![
        array![[0.0]],
        array![[1000.0]], // rejected by the bound
        array![[0.0]],
        array![[60.0]],
        array![[0.0]],
        array![[0.0]],
    ];
    agg.aggregate(&updates, Some(&ids(&["a", "b", "c", "d", "e", "f"])))
        .unwrap();

    for agreeing in ["a", "c", "e", "f"] {
        assert!(
            agg.get_reputation(agreeing) > 0.5,
            "{} agreed with the aggregate and should have gained, got {}",
            agreeing,
            agg.get_reputation(agreeing)
        );
    }
    assert!(
        agg.get_reputation("d") < 0.5,
        "d deviated and should have lost, got {}",
        agg.get_reputation("d")
    );
    assert_eq!(
        agg.get_reputation("b"),
        0.5,
        "b did not participate, so nothing about it was measured"
    );
}

#[test]
fn filtering_works_positionally_without_client_ids() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();

    let updates = vec![array![[1.0]], array![[500.0]], array![[3.0]]];
    let result = agg.aggregate(&updates, None).unwrap();
    assert!(
        (result[[0, 0]] - 2.0).abs() < 1e-6,
        "got {}",
        result[[0, 0]]
    );
}

// ===== Reputation interaction =====

#[test]
fn a_norm_rejection_does_not_move_the_score() {
    // A bound is a participation filter, not a reputation policy. Whether norm
    // violations should also cost reputation is a separate design decision;
    // this pins that it currently does not, in either direction.
    let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();

    let updates = vec![array![[1.0]], array![[1.0]], array![[5000.0]]];
    let names = ids(&["a", "b", "oversized"]);

    let before = agg.get_reputation("oversized");
    for _ in 0..5 {
        agg.aggregate(&updates, Some(&names)).unwrap();
    }

    assert_eq!(
        agg.get_reputation("oversized"),
        before,
        "repeated exclusion must neither penalize nor reward"
    );
    assert!(
        agg.get_reputation("a") > before,
        "accepted clients keep their existing reputation behavior"
    );
}

#[test]
fn reputation_gating_precedes_norm_filtering() {
    // "banned" is both below the threshold and over the bound. Gating runs
    // first, so it is removed before any norm is computed -- and the aggregate
    // is the same either way, which is the point: a client cannot collect two
    // rejections.
    let mut agg = aggregator_with_scores(
        r#""FedAvg""#,
        0.5,
        &[("banned", 0.1), ("good1", 0.9), ("good2", 0.9)],
        Some(10.0),
    );

    let updates = vec![array![[9000.0]], array![[1.0]], array![[3.0]]];
    let result = agg
        .aggregate(&updates, Some(&ids(&["banned", "good1", "good2"])))
        .unwrap();

    assert!(
        (result[[0, 0]] - 2.0).abs() < 1e-6,
        "got {}",
        result[[0, 0]]
    );
}

// ===== Fail closed =====

#[test]
fn every_update_over_the_bound_fails_closed() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(1.0)
        .unwrap();

    let updates = vec![array![[100.0]], array![[200.0]], array![[300.0]]];
    match agg.aggregate(&updates, None) {
        Err(QoraError::AllUpdatesRejected { submitted }) => assert_eq!(submitted, 3),
        other => panic!("expected AllUpdatesRejected, got {:?}", other),
    }
}

#[test]
fn a_mixed_rejection_reports_the_same_generic_error() {
    // One removed by reputation, one by the bound. No per-policy variant could
    // describe this round, which is why the error carries only the count.
    let mut agg = aggregator_with_scores(
        r#""FedAvg""#,
        0.5,
        &[("banned", 0.0), ("oversized", 0.9)],
        Some(10.0),
    );

    let updates = vec![array![[1.0]], array![[5000.0]]];
    match agg.aggregate(&updates, Some(&ids(&["banned", "oversized"]))) {
        Err(QoraError::AllUpdatesRejected { submitted }) => assert_eq!(submitted, 2),
        other => panic!("expected AllUpdatesRejected, got {:?}", other),
    }
}

#[test]
fn no_aggregation_runs_when_everything_is_rejected() {
    // Krum would fail its own quorum check on this cohort (n=3, f=1 needs 5).
    // Filtering short-circuits first, proving the method was never invoked on
    // a reinstated set.
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0)
        .with_norm_bound_filter(1.0)
        .unwrap();

    let updates = vec![array![[100.0]], array![[200.0]], array![[300.0]]];
    assert!(matches!(
        agg.aggregate(&updates, None),
        Err(QoraError::AllUpdatesRejected { submitted: 3 })
    ));
}

#[test]
fn reputation_is_untouched_when_everything_is_rejected() {
    let mut agg = aggregator_with_scores(r#""FedAvg""#, 0.0, &[("a", 0.7), ("b", 0.3)], Some(1.0));
    let updates = vec![array![[100.0]], array![[200.0]]];
    let names = ids(&["a", "b"]);

    for _ in 0..5 {
        assert!(agg.aggregate(&updates, Some(&names)).is_err());
    }

    assert_eq!(agg.get_reputation("a"), 0.7);
    assert_eq!(agg.get_reputation("b"), 0.3);
}

// ===== Method preconditions are evaluated after filtering =====

#[test]
fn krum_valid_before_filtering_can_become_invalid_after() {
    let small = array![[1.0, 1.0]];
    let huge = array![[900.0, 900.0]];

    // n=6 submitted, 5 accepted: still clears 2f+3 = 5 for f=1.
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();
    let six = vec![
        small.clone(),
        array![[1.1, 0.9]],
        array![[0.9, 1.1]],
        array![[1.05, 0.95]],
        array![[1.0, 1.05]],
        huge.clone(),
    ];
    assert!(agg.aggregate(&six, None).is_ok());

    // n=5 submitted, 4 accepted: the reduced cohort is refused rather than
    // repaired by reinstating the excluded client or weakening f.
    let five = vec![
        small.clone(),
        array![[1.1, 0.9]],
        array![[0.9, 1.1]],
        array![[1.05, 0.95]],
        huge,
    ];
    match agg.aggregate(&five, None) {
        Err(QoraError::InsufficientQuorum { needed, actual }) => {
            assert_eq!(needed, 5);
            assert_eq!(actual, 4, "the quorum is judged on the accepted cohort");
        }
        other => panic!("expected InsufficientQuorum, got {:?}", other),
    }
}

#[test]
fn an_explicit_multi_krum_m_is_never_reduced_by_filtering() {
    // n=7, f=1 supports m <= 3. Removing one leaves 6, where the maximum is 2 --
    // so an explicit 3 is refused rather than silently lowered.
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, Some(3)), 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();

    let mut updates: Vec<Array2<f32>> = (0..6).map(|i| array![[1.0 + i as f32 * 0.01]]).collect();
    updates.push(array![[900.0]]);

    match agg.aggregate(&updates, None) {
        Err(QoraError::InvalidMultiKrumSelection {
            clients,
            byzantine,
            selected,
            maximum,
        }) => {
            assert_eq!(clients, 6, "measured against the accepted cohort");
            assert_eq!(byzantine, 1);
            assert_eq!(selected, 3);
            assert_eq!(maximum, 2);
        }
        other => panic!("expected InvalidMultiKrumSelection, got {:?}", other),
    }
}

#[test]
fn a_bare_multi_krum_resolves_from_the_accepted_cohort() {
    // Same cohort, no explicit m: the omitted form caps to min(3, 6-2-2) = 2
    // rather than failing.
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, None), 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();

    let mut updates: Vec<Array2<f32>> = (0..6).map(|i| array![[1.0 + i as f32 * 0.01]]).collect();
    updates.push(array![[900.0]]);

    let result = agg.aggregate(&updates, None).unwrap();
    assert!(
        result[[0, 0]] < 2.0,
        "the excluded outlier must not reach the average, got {}",
        result[[0, 0]]
    );
}
