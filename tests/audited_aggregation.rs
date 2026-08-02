//! The audited aggregation API, as an external caller sees it.
//!
//! `aggregate_with_audit` returns what `aggregate` returns plus the record of
//! what the round decided about each submitted update. The properties pinned
//! here:
//!
//! * The two APIs share one engine, so they agree on both results and errors.
//! * Every submitted update gets exactly one decision, at its original index.
//! * The recorded method parameters are the ones that actually executed.
//! * The record survives the all-rejected failure, which is the round it exists
//!   for.
//! * Records are handed to the caller and not retained.

use ndarray::{array, Array2};
use qora_fl::aggregators::{AggregationMethod, ByzantineAggregator};
use qora_fl::{
    AggregationAuditEntry, AggregationAuditOutcome, AggregationRejectionReason,
    AuditedAggregationMethod, QoraError,
};

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

/// The rejection reason recorded at `index`, or `None` if it was accepted.
fn reason_at(entry: &AggregationAuditEntry, index: usize) -> Option<&AggregationRejectionReason> {
    entry.decisions()[index].decision.rejection_reason()
}

// ===== Successful rounds =====

#[test]
fn a_successful_round_records_every_update_in_submitted_order() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();

    let updates = vec![
        array![[1.0]],
        array![[500.0]], // over the bound
        array![[2.0]],
        array![[3.0]],
    ];
    let result = agg
        .aggregate_with_audit(&updates, Some(&ids(&["a", "b", "c", "d"])))
        .expect("three clients survive");
    let entry = result.audit();

    assert_eq!(entry.outcome(), &AggregationAuditOutcome::Aggregated);
    assert_eq!(entry.submitted_count(), 4);
    assert_eq!(entry.accepted_count(), 3);
    assert_eq!(entry.rejected_count(), 1);
    assert_eq!(
        entry.submitted_count(),
        entry.accepted_count() + entry.rejected_count()
    );

    let indices: Vec<usize> = entry.decisions().iter().map(|d| d.update_index).collect();
    assert_eq!(indices, vec![0, 1, 2, 3], "no renumbering, no reordering");

    let recorded: Vec<Option<&str>> = entry
        .decisions()
        .iter()
        .map(|d| d.client_id.as_deref())
        .collect();
    assert_eq!(recorded, vec![Some("a"), Some("b"), Some("c"), Some("d")]);

    assert!(entry.decisions()[0].decision.is_accepted());
    assert!(matches!(
        reason_at(entry, 1),
        Some(AggregationRejectionReason::NormBoundExceeded { bound, .. }) if *bound == 10.0
    ));
}

#[test]
fn the_recorded_norm_is_the_measured_one() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0)
        .with_norm_bound_filter(5.0)
        .unwrap();

    // Norm is exactly 13.0 (5-12-13 triple), well outside f32 rounding doubt.
    let updates = vec![array![[1.0, 0.0]], array![[5.0, 12.0]], array![[2.0, 0.0]]];
    let result = agg.aggregate_with_audit(&updates, None).unwrap();

    match reason_at(result.audit(), 1) {
        Some(AggregationRejectionReason::NormBoundExceeded { norm, bound }) => {
            assert!((norm - 13.0).abs() < 1e-9, "got {}", norm);
            assert_eq!(*bound, 5.0);
        }
        other => panic!("expected a norm rejection, got {:?}", other),
    }
}

#[test]
fn the_audited_aggregate_equals_the_ordinary_one() {
    let updates = vec![array![[1.0]], array![[500.0]], array![[2.0]], array![[3.0]]];

    let build = || {
        ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
            .with_norm_bound_filter(10.0)
            .unwrap()
    };

    let plain = build().aggregate(&updates, None).unwrap();
    let audited = build().aggregate_with_audit(&updates, None).unwrap();

    assert_eq!(audited.aggregate(), &plain);
}

#[test]
fn a_round_without_client_ids_records_absence_rather_than_positions() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0);
    let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];

    let result = agg.aggregate_with_audit(&updates, None).unwrap();
    assert!(result
        .audit()
        .decisions()
        .iter()
        .all(|d| d.client_id.is_none()));
    assert_eq!(result.audit().accepted_count(), 3);
}

#[test]
fn an_unfiltered_round_accepts_everything() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0);
    let updates = vec![array![[1.0]], array![[5000.0]], array![[3.0]]];

    let result = agg.aggregate_with_audit(&updates, None).unwrap();
    assert_eq!(result.audit().rejected_count(), 0);
    assert!(result
        .audit()
        .decisions()
        .iter()
        .all(|d| d.decision.is_accepted()));
}

// ===== One reason per update =====

#[test]
fn a_reputation_rejected_client_is_not_judged_twice() {
    // "banned" is below the threshold *and* over the bound. Gating runs first
    // and owns the reason; the norm filter never sees it.
    let mut agg = aggregator_with_scores(
        r#""FedAvg""#,
        0.5,
        &[("banned", 0.1), ("good", 0.9), ("oversized", 0.9)],
        Some(10.0),
    );

    let updates = vec![array![[9000.0]], array![[1.0]], array![[5000.0]]];
    let result = agg
        .aggregate_with_audit(&updates, Some(&ids(&["banned", "good", "oversized"])))
        .expect("one client survives");
    let entry = result.audit();

    match reason_at(entry, 0) {
        Some(AggregationRejectionReason::ReputationBelowThreshold { score, threshold }) => {
            assert!((score - 0.1).abs() < 1e-6);
            assert!((threshold - 0.5).abs() < 1e-6);
        }
        other => panic!("gating must own the reason, got {:?}", other),
    }
    assert!(entry.decisions()[1].decision.is_accepted());
    assert!(matches!(
        reason_at(entry, 2),
        Some(AggregationRejectionReason::NormBoundExceeded { .. })
    ));
}

// ===== All rejected =====

#[test]
fn an_all_rejected_round_keeps_every_reason() {
    let mut agg = aggregator_with_scores(
        r#""FedAvg""#,
        0.5,
        &[("banned", 0.0), ("oversized", 0.9)],
        Some(10.0),
    );

    let updates = vec![array![[1.0]], array![[5000.0]]];
    let error = agg
        .aggregate_with_audit(&updates, Some(&ids(&["banned", "oversized"])))
        .expect_err("nothing survives");

    assert!(matches!(
        error.source_error(),
        QoraError::AllUpdatesRejected { submitted: 2 }
    ));

    let entry = error.audit().expect("the all-rejected round is auditable");
    assert_eq!(
        entry.outcome(),
        &AggregationAuditOutcome::AllUpdatesRejected
    );
    assert_eq!(entry.accepted_count(), 0);
    assert_eq!(entry.rejected_count(), 2);
    assert!(matches!(
        reason_at(entry, 0),
        Some(AggregationRejectionReason::ReputationBelowThreshold { .. })
    ));
    assert!(matches!(
        reason_at(entry, 1),
        Some(AggregationRejectionReason::NormBoundExceeded { .. })
    ));
}

#[test]
fn the_ordinary_api_returns_the_same_error_without_the_record() {
    let updates = vec![array![[100.0]], array![[200.0]]];
    let build = || {
        ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
            .with_norm_bound_filter(1.0)
            .unwrap()
    };

    let plain = build().aggregate(&updates, None).unwrap_err();
    let audited = build().aggregate_with_audit(&updates, None).unwrap_err();

    assert_eq!(plain.to_string(), audited.source_error().to_string());
    assert!(audited.audit().is_some(), "only the audited API keeps it");
}

#[test]
fn into_parts_yields_the_error_and_the_record() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(1.0)
        .unwrap();

    let (error, audit) = agg
        .aggregate_with_audit(&[array![[100.0]]], None)
        .unwrap_err()
        .into_parts();

    assert!(matches!(
        error,
        QoraError::AllUpdatesRejected { submitted: 1 }
    ));
    assert_eq!(audit.unwrap().rejected_count(), 1);
}

// ===== Failures the schema cannot describe =====

#[test]
fn a_validation_failure_carries_no_record() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();

    for (updates, ids_arg) in [
        (vec![], None),
        (vec![array![[1.0]], array![[f32::NAN]]], None),
        (vec![array![[1.0, 2.0]], array![[1.0]]], None),
        (vec![array![[1.0]]], Some(ids(&["a", "b"]))),
    ] {
        let error = agg
            .aggregate_with_audit(&updates, ids_arg.as_deref())
            .expect_err("malformed input must fail");
        assert!(
            error.audit().is_none(),
            "no update was individually judged, so there is nothing to record: {}",
            error
        );
    }
}

#[test]
fn an_invalid_configuration_carries_no_record() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Median, 0.0);
    let error = agg
        .aggregate_weighted_with_audit(&[array![[1.0]], array![[2.0]]], None, Some(&[1.0, 2.0]))
        .expect_err("weights are not supported by median");

    assert!(matches!(
        error.source_error(),
        QoraError::WeightsNotSupported { .. }
    ));
    assert!(error.audit().is_none());
}

#[test]
fn a_method_precondition_failure_carries_no_record() {
    // Some candidates survived, so the round is neither "aggregated" nor
    // "all updates rejected" -- the two outcomes schema version 1 has. The
    // typed error is returned on its own rather than with a record that would
    // describe something that did not happen.
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();

    let updates = vec![
        array![[1.0, 1.0]],
        array![[1.1, 0.9]],
        array![[0.9, 1.1]],
        array![[1.05, 0.95]],
        array![[900.0, 900.0]], // excluded, dropping the cohort to 4
    ];

    let error = agg
        .aggregate_with_audit(&updates, None)
        .expect_err("4 accepted clients cannot satisfy 2f+3 = 5");

    assert!(matches!(
        error.source_error(),
        QoraError::InsufficientQuorum {
            needed: 5,
            actual: 4
        }
    ));
    assert!(error.audit().is_none());
}

// ===== Effective method parameters =====

#[test]
fn a_static_trimmed_mean_records_the_fraction_it_applied() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);
    let updates: Vec<Array2<f32>> = (0..5).map(|i| array![[1.0 + i as f32]]).collect();

    let result = agg.aggregate_with_audit(&updates, None).unwrap();
    match result.audit().method() {
        AuditedAggregationMethod::TrimmedMean {
            configured_trim_fraction,
            effective_trim_fraction,
            adaptive,
        } => {
            assert!((configured_trim_fraction - 0.2).abs() < 1e-6);
            assert_eq!(*effective_trim_fraction, Some(*configured_trim_fraction));
            assert!(!adaptive);
        }
        other => panic!("expected trimmed mean, got {:?}", other),
    }
}

#[test]
fn an_adaptive_trimmed_mean_records_the_fraction_that_ran() {
    // Three of the ten known clients sit below the 0.4 suspicion threshold, so
    // the adaptive fraction resolves to 0.30 + 0.05 = 0.35 -- nothing like the
    // configured 0.0. The record must show what executed, not what was set.
    let mut agg = aggregator_with_scores(
        r#""TrimmedMean""#,
        0.0,
        &[
            ("s1", 0.1),
            ("s2", 0.1),
            ("s3", 0.1),
            ("h1", 0.8),
            ("h2", 0.8),
            ("h3", 0.8),
            ("h4", 0.8),
            ("h5", 0.8),
            ("h6", 0.8),
            ("h7", 0.8),
        ],
        None,
    );
    agg.set_adaptive_trim(true);

    let updates: Vec<Array2<f32>> = (0..10).map(|i| array![[1.0 + i as f32]]).collect();
    let result = agg.aggregate_with_audit(&updates, None).unwrap();

    match result.audit().method() {
        AuditedAggregationMethod::TrimmedMean {
            configured_trim_fraction,
            effective_trim_fraction,
            adaptive,
        } => {
            assert!(adaptive);
            assert!((configured_trim_fraction - 0.0).abs() < 1e-6);
            let effective = effective_trim_fraction.expect("a round that ran applied a fraction");
            assert!(
                (effective - 0.35).abs() < 1e-6,
                "expected the resolved 0.35, got {}",
                effective
            );
        }
        other => panic!("expected trimmed mean, got {:?}", other),
    }
}

#[test]
fn krum_records_the_f_it_used() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(2), 0.0);
    let updates: Vec<Array2<f32>> = (0..7).map(|i| array![[1.0 + i as f32 * 0.01]]).collect();

    let result = agg.aggregate_with_audit(&updates, None).unwrap();
    assert_eq!(
        result.audit().method(),
        &AuditedAggregationMethod::Krum { f: 2 }
    );
}

#[test]
fn a_bare_multi_krum_records_the_count_it_resolved() {
    // 7 submitted, one over the bound: 6 accepted, so m caps to min(3, 6-2-2) = 2.
    // The record must show the resolution the accepted cohort produced.
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, None), 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();

    let mut updates: Vec<Array2<f32>> = (0..6).map(|i| array![[1.0 + i as f32 * 0.01]]).collect();
    updates.push(array![[900.0]]);

    let result = agg.aggregate_with_audit(&updates, None).unwrap();
    assert_eq!(
        result.audit().method(),
        &AuditedAggregationMethod::MultiKrum {
            f: 1,
            requested_m: None,
            effective_m: Some(2),
        }
    );
}

#[test]
fn an_explicit_multi_krum_records_the_count_it_was_given() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, Some(2)), 0.0);
    let updates: Vec<Array2<f32>> = (0..7).map(|i| array![[1.0 + i as f32 * 0.01]]).collect();

    let result = agg.aggregate_with_audit(&updates, None).unwrap();
    assert_eq!(
        result.audit().method(),
        &AuditedAggregationMethod::MultiKrum {
            f: 1,
            requested_m: Some(2),
            effective_m: Some(2),
        }
    );
}

#[test]
fn a_refused_round_still_names_its_configured_method() {
    let mut agg = ByzantineAggregator::new(AggregationMethod::Krum(1), 0.0)
        .with_norm_bound_filter(1.0)
        .unwrap();

    let entry_owner = agg
        .aggregate_with_audit(&[array![[100.0]], array![[200.0]]], None)
        .unwrap_err();
    let entry = entry_owner.audit().expect("all-rejected is auditable");

    assert_eq!(entry.method(), &AuditedAggregationMethod::Krum { f: 1 });
}

#[test]
fn a_refused_round_records_no_effective_trim_fraction() {
    // The configured fraction and the adaptive flag are properties of the
    // aggregator and are still reported; the effective fraction is not, because
    // no trimming happened.
    let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2)
        .with_norm_bound_filter(1.0)
        .unwrap();
    agg.set_adaptive_trim(true);

    let error = agg
        .aggregate_with_audit(&[array![[100.0]], array![[200.0]]], None)
        .unwrap_err();

    match error.audit().expect("all-rejected is auditable").method() {
        AuditedAggregationMethod::TrimmedMean {
            configured_trim_fraction,
            effective_trim_fraction,
            adaptive,
        } => {
            assert!((configured_trim_fraction - 0.2).abs() < 1e-6);
            assert_eq!(
                *effective_trim_fraction, None,
                "no trimming ran, so no fraction was applied"
            );
            assert!(adaptive);
        }
        other => panic!("expected trimmed mean, got {:?}", other),
    }
}

#[test]
fn a_refused_bare_multi_krum_records_no_effective_m() {
    // The case the Option exists for. A bare `m` resolves against the accepted
    // cohort, and nothing was accepted -- so there is no resolution to report,
    // not even the uncapped default the configuration starts from.
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, None), 0.0)
        .with_norm_bound_filter(1.0)
        .unwrap();

    let updates: Vec<Array2<f32>> = (0..7).map(|_| array![[900.0]]).collect();
    let error = agg.aggregate_with_audit(&updates, None).unwrap_err();

    assert_eq!(
        error.audit().expect("all-rejected is auditable").method(),
        &AuditedAggregationMethod::MultiKrum {
            f: 1,
            requested_m: None,
            effective_m: None,
        }
    );
}

#[test]
fn a_refused_explicit_multi_krum_keeps_the_request_but_not_a_result() {
    // Asking for m=2 is a fact about the configuration and survives; it is not
    // evidence that two vectors were ever selected.
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, Some(2)), 0.0)
        .with_norm_bound_filter(1.0)
        .unwrap();

    let updates: Vec<Array2<f32>> = (0..7).map(|_| array![[900.0]]).collect();
    let error = agg.aggregate_with_audit(&updates, None).unwrap_err();

    assert_eq!(
        error.audit().expect("all-rejected is auditable").method(),
        &AuditedAggregationMethod::MultiKrum {
            f: 1,
            requested_m: Some(2),
            effective_m: None,
        }
    );
}

#[test]
fn a_refused_round_record_still_round_trips() {
    // The `null` effective values must survive serialization like any other
    // part of the record.
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, None), 0.0)
        .with_norm_bound_filter(1.0)
        .unwrap();

    let updates: Vec<Array2<f32>> = (0..7).map(|_| array![[900.0]]).collect();
    let error = agg.aggregate_with_audit(&updates, None).unwrap_err();
    let entry = error.audit().expect("all-rejected is auditable");

    let json = serde_json::to_string(entry).unwrap();
    assert!(
        json.contains(r#""effective_m":null"#),
        "the absent resolution must be explicit on the wire: {}",
        json
    );

    let restored: AggregationAuditEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(&restored, entry);
}

// ===== Serialization =====

#[test]
fn a_returned_record_round_trips_through_json() {
    let mut agg = aggregator_with_scores(
        r#""TrimmedMean""#,
        0.5,
        &[("banned", 0.1), ("a", 0.9), ("b", 0.9), ("c", 0.9)],
        Some(10.0),
    );

    let updates = vec![
        array![[9000.0]], // banned by reputation
        array![[1.0]],
        array![[500.0]], // over the bound
        array![[2.0]],
    ];
    let result = agg
        .aggregate_with_audit(&updates, Some(&ids(&["banned", "a", "b", "c"])))
        .unwrap();

    let json = serde_json::to_string(result.audit()).unwrap();
    let restored: AggregationAuditEntry = serde_json::from_str(&json).unwrap();

    assert_eq!(&restored, result.audit());
    assert_eq!(restored.schema_version(), 1);
}

// ===== Ownership =====

#[test]
fn records_are_not_retained_by_the_aggregator() {
    // Nothing accumulates: the serialized aggregator after several audited
    // rounds is identical to one that only ever aggregated.
    let updates = vec![array![[1.0]], array![[500.0]], array![[2.0]]];

    let mut audited = ByzantineAggregator::new(AggregationMethod::Median, 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();
    let mut plain = ByzantineAggregator::new(AggregationMethod::Median, 0.0)
        .with_norm_bound_filter(10.0)
        .unwrap();

    for _ in 0..3 {
        audited.aggregate_with_audit(&updates, None).unwrap();
        plain.aggregate(&updates, None).unwrap();
    }

    assert_eq!(
        serde_json::to_string(&audited).unwrap(),
        serde_json::to_string(&plain).unwrap()
    );
}
