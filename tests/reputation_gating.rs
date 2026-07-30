//! Reputation participation gating fails closed.
//!
//! Gating previously fell back to the full cohort when no client cleared the
//! ban threshold, so a banned client was reinstated precisely when reputation
//! distrusted everyone. The rule pinned here is narrow:
//!
//! > A client rejected by the configured reputation policy is never silently
//! > restored merely because every other client was also rejected.
//!
//! These tests cover only that policy. How reputation scores are *calculated*
//! is unchanged and is not exercised here.

use ndarray::{array, Array2};
use qora_fl::aggregators::{AggregationMethod, ByzantineAggregator};
use qora_fl::QoraError;

/// Build an aggregator with exact reputation scores.
///
/// `ByzantineAggregator` exposes no score setter, and driving scores through
/// repeated rounds would couple these tests to the reputation arithmetic they
/// are meant to leave alone. Deserialization sets the state directly.
fn aggregator_with_scores(
    method: &str,
    ban_threshold: f32,
    scores: &[(&str, f32)],
) -> ByzantineAggregator {
    let entries: Vec<String> = scores
        .iter()
        .map(|(id, score)| format!(r#""{}":{}"#, id, score))
        .collect();
    let json = format!(
        r#"{{"method":{},"trim_fraction":0.0,"reputation":{{{}}},"ban_threshold":{},"adaptive_trim":false}}"#,
        method,
        entries.join(","),
        ban_threshold
    );
    serde_json::from_str(&json).expect("aggregator fixture must deserialize")
}

fn ids(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

// ===== The core policy =====

#[test]
fn all_banned_clients_return_typed_error() {
    let mut agg = aggregator_with_scores(
        r#""FedAvg""#,
        0.5,
        &[("a", 0.1), ("b", 0.05), ("c", 0.0), ("d", 0.2)],
    );

    let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]], array![[4.0]]];

    match agg.aggregate(&updates, Some(&ids(&["a", "b", "c", "d"]))) {
        Err(QoraError::AllUpdatesRejected {
            total,
            rejected,
            threshold,
        }) => {
            assert_eq!(total, 4, "total should count every submitted update");
            assert_eq!(rejected, 4, "all four were below the threshold");
            assert!((threshold - 0.5).abs() < 1e-6);
        }
        other => panic!("expected AllUpdatesRejected, got {:?}", other),
    }
}

#[test]
fn single_accepted_client_is_not_restored_into_the_full_group() {
    // One client clears the threshold; the other two are far below it. Values
    // are chosen so that any restoration of the banned updates would be
    // unmistakable in the result.
    let mut agg = aggregator_with_scores(
        r#""FedAvg""#,
        0.5,
        &[("good", 0.9), ("bad1", 0.0), ("bad2", 0.0)],
    );

    let updates = vec![array![[10.0]], array![[1000.0]], array![[-1000.0]]];
    let result = agg
        .aggregate(&updates, Some(&ids(&["good", "bad1", "bad2"])))
        .expect("a partially accepted cohort must still aggregate");

    assert!(
        (result[[0, 0]] - 10.0).abs() < 1e-6,
        "result should come from the accepted client alone, got {}",
        result[[0, 0]]
    );
}

#[test]
fn all_rejected_takes_precedence_over_method_execution() {
    // Krum would fail its own quorum check on this cohort (n=3, f=1 needs 5).
    // Gating runs first, so the caller hears about the policy rejection --
    // proving the method was never invoked on the reinstated set.
    let mut agg =
        aggregator_with_scores(r#"{"Krum":1}"#, 0.5, &[("a", 0.1), ("b", 0.1), ("c", 0.1)]);

    let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];

    assert!(
        matches!(
            agg.aggregate(&updates, Some(&ids(&["a", "b", "c"]))),
            Err(QoraError::AllUpdatesRejected { .. })
        ),
        "gating must short-circuit before Krum's quorum check"
    );
}

// ===== Reputation state must not move on the error path =====

#[test]
fn reputation_state_is_unchanged_when_all_updates_are_rejected() {
    let names = ["a", "b", "c"];
    let mut agg =
        aggregator_with_scores(r#""FedAvg""#, 0.5, &[("a", 0.1), ("b", 0.25), ("c", 0.0)]);

    let before: Vec<f32> = names.iter().map(|n| agg.get_reputation(n)).collect();

    let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];
    assert!(matches!(
        agg.aggregate(&updates, Some(&ids(&names))),
        Err(QoraError::AllUpdatesRejected { .. })
    ));

    let after: Vec<f32> = names.iter().map(|n| agg.get_reputation(n)).collect();
    assert_eq!(
        before, after,
        "a round that failed the gate must not move the scores that failed it"
    );
}

#[test]
fn repeated_rejection_does_not_compound_penalties() {
    // A caller retrying a rejected round must not drive scores lower each
    // time; the gate reads state, it does not write it.
    let mut agg = aggregator_with_scores(r#""FedAvg""#, 0.5, &[("a", 0.4), ("b", 0.4)]);
    let updates = vec![array![[1.0]], array![[500.0]]];

    let before = agg.get_reputation("a");
    for _ in 0..5 {
        assert!(agg.aggregate(&updates, Some(&ids(&["a", "b"]))).is_err());
    }
    assert_eq!(before, agg.get_reputation("a"));
}

// ===== Validation precedence: malformed input is never masked =====

#[test]
fn client_id_mismatch_outranks_gating() {
    let mut agg = aggregator_with_scores(r#""Median""#, 0.5, &[("a", 0.0), ("b", 0.0), ("c", 0.0)]);

    let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];

    assert!(
        matches!(
            agg.aggregate(&updates, Some(&ids(&["a", "b"]))),
            Err(QoraError::ClientIdCountMismatch {
                updates: 3,
                client_ids: 2
            })
        ),
        "a count mismatch must not be reported as a policy rejection"
    );
}

#[test]
fn non_finite_updates_outrank_gating() {
    // Every client here would also be banned. Malformed input must still win,
    // so a rejected cohort cannot be used to conceal a poisoned update.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut agg =
            aggregator_with_scores(r#""Median""#, 0.5, &[("a", 0.0), ("b", 0.0), ("c", 0.0)]);

        let updates = vec![array![[1.0]], array![[bad]], array![[3.0]]];

        assert!(
            matches!(
                agg.aggregate(&updates, Some(&ids(&["a", "b", "c"]))),
                Err(QoraError::NonFiniteValue {
                    update_index: 1,
                    ..
                })
            ),
            "{} should be reported as NonFiniteValue, not AllUpdatesRejected",
            bad
        );
    }
}

#[test]
fn empty_updates_outrank_gating() {
    let mut agg = aggregator_with_scores(r#""FedAvg""#, 0.5, &[("a", 0.0)]);
    let updates: Vec<Array2<f32>> = vec![];

    assert!(matches!(
        agg.aggregate(&updates, Some(&[])),
        Err(QoraError::EmptyUpdates)
    ));
}

// ===== Gating applies only where it can =====

#[test]
fn no_client_ids_means_no_gating() {
    // Reputation holds banned entries, but without IDs no update can be
    // associated with one, so gating cannot apply.
    let mut agg = aggregator_with_scores(
        r#""FedAvg""#,
        0.5,
        &[("someone_else", 0.0), ("another", 0.0)],
    );

    let updates = vec![array![[1.0]], array![[3.0]]];
    let result = agg
        .aggregate(&updates, None)
        .expect("aggregation without IDs is ungated");
    assert!((result[[0, 0]] - 2.0).abs() < 1e-6);
}

#[test]
fn disabled_gating_preserves_normal_behavior() {
    // ban_threshold = 0.0 disables the gate. Callers not using reputation at
    // all must be unaffected by the fail-closed change.
    let mut agg = aggregator_with_scores(r#""FedAvg""#, 0.0, &[("a", 0.0), ("b", 0.0)]);

    let updates = vec![array![[1.0]], array![[3.0]]];
    let result = agg
        .aggregate(&updates, Some(&ids(&["a", "b"])))
        .expect("gating is off, so scores of 0.0 are irrelevant");
    assert!((result[[0, 0]] - 2.0).abs() < 1e-6);
}

#[test]
fn default_constructor_never_gates() {
    // `new` leaves ban_threshold at 0.0, so the fail-closed path is
    // unreachable for the common constructor.
    let mut agg = ByzantineAggregator::new(AggregationMethod::FedAvg, 0.0);
    let updates = vec![array![[1.0]], array![[3.0]]];
    assert!(agg.aggregate(&updates, Some(&ids(&["a", "b"]))).is_ok());
}

// ===== Error message =====

#[test]
fn error_message_names_the_counts_and_threshold() {
    let err = QoraError::AllUpdatesRejected {
        total: 4,
        rejected: 4,
        threshold: 0.5,
    };
    let msg = err.to_string();

    assert!(msg.contains('4'), "should name the counts: {}", msg);
    assert!(msg.contains("0.5"), "should name the threshold: {}", msg);
    assert!(
        msg.contains("reputation gating"),
        "should attribute the rejection to gating: {}",
        msg
    );
}
