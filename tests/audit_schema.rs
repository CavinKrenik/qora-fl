//! The audit schema as an external consumer sees it.
//!
//! Everything here goes through `qora_fl::` paths only. A type that exists but
//! was never exported would compile fine inside the crate and fail here.
//!
//! Entries are obtained by deserializing: the constructor is crate-visible
//! because no aggregation path produces entries yet, so deserialization is the
//! only external route in -- and it enforces the same invariants.

use qora_fl::{
    AggregationAuditDecision, AggregationAuditEntry, AggregationAuditOutcome, AggregationDecision,
    AggregationRejectionReason, AuditedAggregationMethod, AGGREGATION_AUDIT_SCHEMA_VERSION,
};

/// Static trimmed mean, serialized.
const STATIC_TRIM: &str = r#"{"trimmed_mean":{"configured_trim_fraction":0.25,"effective_trim_fraction":0.25,"adaptive":false}}"#;
/// Adaptive trimmed mean whose round value differs from its baseline.
const ADAPTIVE_TRIM: &str = r#"{"trimmed_mean":{"configured_trim_fraction":0.125,"effective_trim_fraction":0.375,"adaptive":true}}"#;

/// Build a JSON entry the way a caller's stored record would look.
fn entry_json(method: &str, decisions: &str, outcome: &str) -> String {
    format!(
        r#"{{"schema_version":{},"method":{},"decisions":[{}],"outcome":"{}"}}"#,
        AGGREGATION_AUDIT_SCHEMA_VERSION, method, decisions, outcome
    )
}

fn accepted(index: usize, id: Option<&str>) -> String {
    let id = id.map_or("null".to_string(), |s| format!(r#""{}""#, s));
    format!(
        r#"{{"update_index":{},"client_id":{},"decision":"accepted"}}"#,
        index, id
    )
}

fn reputation_rejected(index: usize, id: Option<&str>, score: f32, threshold: f32) -> String {
    let id = id.map_or("null".to_string(), |s| format!(r#""{}""#, s));
    format!(
        r#"{{"update_index":{},"client_id":{},"decision":{{"rejected":{{"reputation_below_threshold":{{"score":{},"threshold":{}}}}}}}}}"#,
        index, id, score, threshold
    )
}

fn norm_rejected(index: usize, id: Option<&str>, norm: f64, bound: f32) -> String {
    let id = id.map_or("null".to_string(), |s| format!(r#""{}""#, s));
    format!(
        r#"{{"update_index":{},"client_id":{},"decision":{{"rejected":{{"norm_bound_exceeded":{{"norm":{},"bound":{}}}}}}}}}"#,
        index, id, norm, bound
    )
}

fn parse(json: &str) -> AggregationAuditEntry {
    serde_json::from_str(json).unwrap_or_else(|e| panic!("entry must deserialize: {}\n{}", e, json))
}

// ===== The five scenarios the schema must be able to represent =====
//
// Walked through before the schema is published, so that wiring norm filtering
// in later cannot discover the record shape is inadequate.

#[test]
fn scenario_reputation_rejection_only() {
    // 3 submitted, 2 accepted, 1 reputation-rejected, aggregate produced.
    let entry = parse(&entry_json(
        r#""median""#,
        &[
            accepted(0, Some("a")),
            reputation_rejected(1, Some("b"), 0.125, 0.25),
            accepted(2, Some("c")),
        ]
        .join(","),
        "aggregated",
    ));

    assert_eq!(entry.submitted_count(), 3);
    assert_eq!(entry.accepted_count(), 2);
    assert_eq!(entry.rejected_count(), 1);
    assert_eq!(entry.outcome(), &AggregationAuditOutcome::Aggregated);

    match entry.decisions()[1].decision.rejection_reason() {
        Some(AggregationRejectionReason::ReputationBelowThreshold { score, threshold }) => {
            assert_eq!(*score, 0.125);
            assert_eq!(*threshold, 0.25);
        }
        other => panic!("expected a reputation rejection, got {:?}", other),
    }
}

#[test]
fn scenario_norm_rejection_only() {
    // 3 submitted, 2 accepted, 1 over the bound, aggregate produced. The norm
    // must survive as f64 -- this value is not representable in f32.
    let norm = 1.414_213_562_373e20_f64;
    let entry = parse(&entry_json(
        ADAPTIVE_TRIM,
        &[
            accepted(0, Some("a")),
            norm_rejected(1, Some("b"), norm, 10.0),
            accepted(2, Some("c")),
        ]
        .join(","),
        "aggregated",
    ));

    match entry.decisions()[1].decision.rejection_reason() {
        Some(AggregationRejectionReason::NormBoundExceeded { norm: n, bound }) => {
            assert_eq!(*n, norm);
            assert_eq!(*bound, 10.0);
        }
        other => panic!("expected a norm rejection, got {:?}", other),
    }
}

#[test]
fn scenario_both_filters_preserve_original_indices() {
    // 4 submitted: index 1 reputation-rejected, index 2 norm-rejected, 0 and 3
    // accepted. The surviving indices must stay 0 and 3, not be renumbered to
    // 0 and 1.
    let entry = parse(&entry_json(
        r#"{"krum":{"f":1}}"#,
        &[
            accepted(0, Some("a")),
            reputation_rejected(1, Some("b"), 0.0, 0.5),
            norm_rejected(2, Some("c"), 500.0, 10.0),
            accepted(3, Some("d")),
        ]
        .join(","),
        "aggregated",
    ));

    assert_eq!(entry.submitted_count(), 4);
    assert_eq!(entry.accepted_count(), 2);

    let accepted_indices: Vec<usize> = entry
        .decisions()
        .iter()
        .filter(|d| d.decision.is_accepted())
        .map(|d| d.update_index)
        .collect();
    assert_eq!(
        accepted_indices,
        vec![0, 3],
        "positions must not be renumbered"
    );

    // Both rejection kinds coexist in one entry.
    let reasons: Vec<_> = entry
        .decisions()
        .iter()
        .filter_map(|d| d.decision.rejection_reason())
        .collect();
    assert!(matches!(
        reasons[0],
        AggregationRejectionReason::ReputationBelowThreshold { .. }
    ));
    assert!(matches!(
        reasons[1],
        AggregationRejectionReason::NormBoundExceeded { .. }
    ));
}

#[test]
fn scenario_all_rejected_produces_no_aggregate() {
    // 2 reputation-rejected, 1 norm-rejected, nothing aggregated. The entry
    // must say so without inventing a result.
    let entry = parse(&entry_json(
        r#""median""#,
        &[
            reputation_rejected(0, Some("a"), 0.0, 0.5),
            reputation_rejected(1, Some("b"), 0.125, 0.5),
            norm_rejected(2, Some("c"), 1e9, 1.0),
        ]
        .join(","),
        "all_updates_rejected",
    ));

    assert_eq!(entry.submitted_count(), 3);
    assert_eq!(entry.accepted_count(), 0);
    assert_eq!(entry.rejected_count(), 3);
    assert_eq!(
        entry.outcome(),
        &AggregationAuditOutcome::AllUpdatesRejected
    );
}

#[test]
fn scenario_positional_updates_without_client_ids() {
    // 3 positional updates, 1 rejected, no identities available.
    let entry = parse(&entry_json(
        r#""fedavg""#,
        &[
            accepted(0, None),
            norm_rejected(1, None, 42.0, 1.0),
            accepted(2, None),
        ]
        .join(","),
        "aggregated",
    ));

    assert!(entry.decisions().iter().all(|d| d.client_id.is_none()));
    assert_eq!(entry.rejected_count(), 1);
    assert_eq!(entry.decisions()[1].update_index, 1);
}

// ===== Public surface =====

#[test]
fn method_parameters_survive_the_public_boundary() {
    // Bare and explicit Multi-Krum must stay distinguishable: they are
    // different requests, and an audit that flattened both to "multi_krum"
    // would lose the distinction the aggregator acts on.
    let bare = parse(&entry_json(
        r#"{"multi_krum":{"f":1,"requested_m":null,"effective_m":2}}"#,
        &accepted(0, None),
        "aggregated",
    ));
    let explicit = parse(&entry_json(
        r#"{"multi_krum":{"f":1,"requested_m":3,"effective_m":3}}"#,
        &accepted(0, None),
        "aggregated",
    ));

    assert_eq!(
        bare.method(),
        &AuditedAggregationMethod::MultiKrum {
            f: 1,
            requested_m: None,
            effective_m: 2,
        }
    );
    assert_eq!(
        explicit.method(),
        &AuditedAggregationMethod::MultiKrum {
            f: 1,
            requested_m: Some(3),
            effective_m: 3,
        }
    );
    assert_ne!(bare.method(), explicit.method());
}

#[test]
fn decisions_are_constructible_and_matchable_externally() {
    // The decision types are usable from outside without the entry
    // constructor, which callers need in order to inspect records.
    let accepted = AggregationAuditDecision::accepted(0, Some("a".into()));
    assert!(accepted.decision.is_accepted());
    assert!(accepted.decision.rejection_reason().is_none());

    let rejected = AggregationAuditDecision::rejected(
        1,
        None,
        AggregationRejectionReason::NormBoundExceeded {
            norm: 12.0,
            bound: 1.0,
        },
    );
    assert!(!rejected.decision.is_accepted());

    // Non-exhaustive enums require a wildcard arm, by design.
    let described = match rejected.decision {
        AggregationDecision::Accepted => "accepted",
        AggregationDecision::Rejected(ref reason) => match reason {
            AggregationRejectionReason::ReputationBelowThreshold { .. } => "reputation",
            AggregationRejectionReason::NormBoundExceeded { .. } => "norm",
            _ => "unrecognized",
        },
    };
    assert_eq!(described, "norm");
}

#[test]
fn entries_round_trip_through_the_public_api() {
    let json = entry_json(
        r#""median""#,
        &[
            accepted(0, Some("a")),
            reputation_rejected(1, None, 0.0, 0.5),
        ]
        .join(","),
        "aggregated",
    );
    let entry = parse(&json);

    let reserialized = serde_json::to_string(&entry).expect("entry must serialize");
    let restored: AggregationAuditEntry =
        serde_json::from_str(&reserialized).expect("round trip must succeed");

    assert_eq!(restored, entry);
    assert_eq!(restored.schema_version(), AGGREGATION_AUDIT_SCHEMA_VERSION);
}

#[test]
fn impossible_entries_cannot_be_deserialized() {
    // The invariants are not merely documented; deserialization is the only
    // external construction route and it enforces them.
    let contradictory = entry_json(r#""median""#, &accepted(0, None), "all_updates_rejected");
    assert!(serde_json::from_str::<AggregationAuditEntry>(&contradictory).is_err());

    let gap = entry_json(
        r#""median""#,
        &[accepted(0, None), accepted(2, None)].join(","),
        "aggregated",
    );
    assert!(serde_json::from_str::<AggregationAuditEntry>(&gap).is_err());

    let bad_score = entry_json(
        r#""median""#,
        &reputation_rejected(0, None, 5.0, 0.2),
        "all_updates_rejected",
    );
    assert!(serde_json::from_str::<AggregationAuditEntry>(&bad_score).is_err());
}
