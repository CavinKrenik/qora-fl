//! Multi-Krum selection-bound enforcement.
//!
//! Multi-Krum documented `m <= n - 2f - 2` (Blanchard et al., 2017) but
//! enforced only `m.max(1).min(n)`, so an oversized `m` produced a
//! normal-looking aggregate carrying no Byzantine guarantee. These tests pin
//! the enforced contract:
//!
//! * `n >= 2f + 3` and `1 <= m <= n - 2f - 2`, or the round is refused.
//! * An **omitted** `m` is capped to `min(3, n - 2f - 2)`.
//! * An **explicit** `m` is honored exactly or refused -- never rewritten.

use ndarray::{array, Array2};
use qora_fl::aggregators::krum::{aggregate_multi_krum_bfp16, Bfp16Vec};
use qora_fl::aggregators::AggregationMethod;
use qora_fl::{ByzantineAggregator, QoraError};

/// Spread-out updates so that the mean of the `k` lowest-scoring vectors is
/// distinct for each `k` -- otherwise a test cannot tell m=1 from m=3.
fn spread_updates(n: usize) -> Vec<Array2<f32>> {
    (0..n)
        .map(|i| array![[1.0 + i as f32 * 1.7]])
        .collect::<Vec<_>>()
}

/// Mean of the `m` lowest-Krum-score updates, computed through the low-level
/// API so the expectation is independent of the aggregator's own choice of m.
fn mean_of_selection(updates: &[Array2<f32>], f: usize, m: usize) -> f32 {
    let vecs: Vec<Bfp16Vec> = updates
        .iter()
        .map(|u| Bfp16Vec::from_f32_slice(&u.iter().copied().collect::<Vec<f32>>()))
        .collect();
    let idx = aggregate_multi_krum_bfp16(&vecs, f, m)
        .unwrap_or_else(|| panic!("low-level rejected f={} m={}", f, m));
    idx.iter().map(|&i| updates[i][[0, 0]]).sum::<f32>() / m as f32
}

fn aggregate_bare(updates: &[Array2<f32>], f: usize) -> f32 {
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(f, None), 0.0);
    agg.aggregate(updates, None)
        .expect("bare form must succeed")[[0, 0]]
}

// ===== Bare method: m capped to min(3, n - 2f - 2) =====

#[test]
fn bare_multi_krum_selects_one_at_five_clients() {
    // n=5, f=1 -> max safe m = 1, so the cap yields m=1.
    let updates = spread_updates(5);
    let got = aggregate_bare(&updates, 1);

    assert!(
        (got - mean_of_selection(&updates, 1, 1)).abs() < 1e-4,
        "n=5 should select exactly 1 vector, got {}",
        got
    );
    // m=1 means no averaging at all: the result is verbatim an input.
    assert!(
        updates.iter().any(|u| (u[[0, 0]] - got).abs() < 1e-4),
        "m=1 result {} should be one of the inputs verbatim",
        got
    );
}

#[test]
fn bare_multi_krum_selects_two_at_six_clients() {
    // n=6, f=1 -> max safe m = 2.
    let updates = spread_updates(6);
    let got = aggregate_bare(&updates, 1);

    assert!(
        (got - mean_of_selection(&updates, 1, 2)).abs() < 1e-4,
        "n=6 should select exactly 2 vectors, got {}",
        got
    );
    assert!(
        (got - mean_of_selection(&updates, 1, 1)).abs() > 1e-3,
        "n=6 must not fall back to a single selection"
    );
}

#[test]
fn bare_multi_krum_selects_three_at_seven_clients() {
    // n=7, f=1 -> max safe m = 3, where the cap stops: the historical
    // default of 3 is preserved for every cohort large enough to support it.
    let updates = spread_updates(7);
    let got = aggregate_bare(&updates, 1);

    assert!(
        (got - mean_of_selection(&updates, 1, 3)).abs() < 1e-4,
        "n=7 should select exactly 3 vectors, got {}",
        got
    );
}

#[test]
fn bare_multi_krum_stays_at_three_for_large_cohorts() {
    // The cap is min(3, max_m), not max_m: a 20-client round still selects 3.
    let updates = spread_updates(20);
    let got = aggregate_bare(&updates, 1);

    assert!(
        (got - mean_of_selection(&updates, 1, 3)).abs() < 1e-4,
        "n=20 should still select 3, got {}",
        got
    );
    assert!(
        (got - mean_of_selection(&updates, 1, 4)).abs() > 1e-3,
        "the omitted-m cap must not grow with n"
    );
}

// ===== Explicit method: honored exactly, or refused =====

#[test]
fn explicit_m_above_safe_maximum_returns_typed_error() {
    // n=5, f=1 -> max safe m = 1; the old default of 3 is now refused.
    let updates = spread_updates(5);
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, Some(3)), 0.0);

    match agg.aggregate(&updates, None) {
        Err(QoraError::InvalidMultiKrumSelection {
            clients,
            byzantine,
            selected,
            maximum,
        }) => {
            assert_eq!(clients, 5);
            assert_eq!(byzantine, 1);
            assert_eq!(selected, 3);
            assert_eq!(maximum, 1);
        }
        other => panic!("expected InvalidMultiKrumSelection, got {:?}", other),
    }
}

#[test]
fn explicit_m_of_zero_is_rejected() {
    let updates = spread_updates(7);
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, Some(0)), 0.0);

    assert!(
        matches!(
            agg.aggregate(&updates, None),
            Err(QoraError::InvalidMultiKrumSelection { selected: 0, .. })
        ),
        "m=0 selects nothing and cannot be averaged"
    );
}

#[test]
fn explicit_m_at_exactly_the_safe_maximum_is_accepted() {
    // n=7, f=1 -> max safe m = 3. Boundary must be inclusive.
    let updates = spread_updates(7);
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, Some(3)), 0.0);
    assert!(agg.aggregate(&updates, None).is_ok());

    // One past it is not.
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, Some(4)), 0.0);
    assert!(matches!(
        agg.aggregate(&updates, None),
        Err(QoraError::InvalidMultiKrumSelection {
            selected: 4,
            maximum: 3,
            ..
        })
    ));
}

#[test]
fn explicit_m_is_never_silently_rewritten() {
    // The whole point of the Option: an explicit request that cannot be
    // honored is an error, not a quietly reduced selection.
    let updates = spread_updates(6);

    let mut explicit = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, Some(3)), 0.0);
    assert!(
        explicit.aggregate(&updates, None).is_err(),
        "explicit m=3 at n=6 (max 2) must fail rather than degrade to m=2"
    );

    // The bare form over the same cohort does degrade, and succeeds.
    let mut bare = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, None), 0.0);
    assert!(bare.aggregate(&updates, None).is_ok());
}

// ===== Quorum outranks the selection bound =====

#[test]
fn quorum_failure_is_reported_before_selection_bound() {
    // n=6, f=2 -> quorum needs 7. Both conditions fail; the caller needs to
    // hear about n, since no m would be valid at this cohort size.
    let updates = spread_updates(6);
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(2, Some(3)), 0.0);

    match agg.aggregate(&updates, None) {
        Err(QoraError::InsufficientQuorum { needed, actual }) => {
            assert_eq!(needed, 7);
            assert_eq!(actual, 6);
        }
        other => panic!("expected InsufficientQuorum, got {:?}", other),
    }
}

#[test]
fn bare_form_still_fails_closed_below_quorum() {
    // Capping m must not paper over a quorum shortfall.
    let updates = spread_updates(4);
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, None), 0.0);

    assert!(matches!(
        agg.aggregate(&updates, None),
        Err(QoraError::InsufficientQuorum {
            needed: 5,
            actual: 4
        })
    ));
}

// ===== Low-level API =====

#[test]
fn low_level_returns_none_for_unsafe_m() {
    let vecs: Vec<Bfp16Vec> = (0..7)
        .map(|i| Bfp16Vec::from_f32_slice(&[i as f32]))
        .collect();

    // n=7, f=1 -> max safe m = 3.
    assert!(aggregate_multi_krum_bfp16(&vecs, 1, 0).is_none(), "m=0");
    assert!(aggregate_multi_krum_bfp16(&vecs, 1, 4).is_none(), "m > max");
    assert!(
        aggregate_multi_krum_bfp16(&vecs, 1, usize::MAX).is_none(),
        "m must not overflow into acceptance"
    );
    assert!(
        aggregate_multi_krum_bfp16(&vecs, 1, 3).is_some(),
        "m == max"
    );

    // Below quorum, nothing is selectable at any m.
    let few: Vec<Bfp16Vec> = (0..4)
        .map(|i| Bfp16Vec::from_f32_slice(&[i as f32]))
        .collect();
    assert!(aggregate_multi_krum_bfp16(&few, 1, 1).is_none());
}

// ===== Serialization =====

#[test]
fn both_forms_survive_serde_roundtrip() {
    for method in [
        AggregationMethod::MultiKrum(1, None),
        AggregationMethod::MultiKrum(1, Some(2)),
    ] {
        let json = serde_json::to_string(&method).expect("serialize");
        let restored: AggregationMethod = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            restored, method,
            "omitted and explicit m must stay distinguishable across a round trip"
        );
    }
}

/// A 0.3.1 payload predates the `Option` and must still load.
///
/// `Option<usize>` serializes untagged, so `[f, m]` remains valid JSON for the
/// new shape and restores as an *explicit* `m` -- which is the correct reading
/// of a configuration that was explicit when it was written.
#[test]
fn pre_option_json_still_deserializes_as_explicit_m() {
    let restored: AggregationMethod =
        serde_json::from_str(r#"{"MultiKrum":[1,3]}"#).expect("0.3.1 payload must still load");
    assert_eq!(restored, AggregationMethod::MultiKrum(1, Some(3)));

    // And the explicit form still round-trips to that same shape, so a 0.4.0
    // writer stays readable by anything parsing the old layout.
    assert_eq!(
        serde_json::to_string(&AggregationMethod::MultiKrum(1, Some(3))).unwrap(),
        r#"{"MultiKrum":[1,3]}"#
    );

    // The omitted form is the genuinely new encoding.
    assert_eq!(
        serde_json::to_string(&AggregationMethod::MultiKrum(1, None)).unwrap(),
        r#"{"MultiKrum":[1,null]}"#
    );
}

#[test]
fn aggregator_with_bare_form_survives_serde_roundtrip() {
    let updates = spread_updates(6);
    let mut agg = ByzantineAggregator::new(AggregationMethod::MultiKrum(1, None), 0.0);
    let ids: Vec<String> = (0..6).map(|i| format!("c{}", i)).collect();
    let before = agg.aggregate(&updates, Some(&ids)).unwrap();

    let json = serde_json::to_string(&agg).expect("serialize");
    let mut restored: ByzantineAggregator = serde_json::from_str(&json).expect("deserialize");
    let after = restored.aggregate(&updates, Some(&ids)).unwrap();

    assert_eq!(
        before, after,
        "a restored bare-form aggregator must still cap m the same way"
    );
}
