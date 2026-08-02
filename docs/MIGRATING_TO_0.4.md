# Migrating to 0.4.0

0.4.0 is a breaking release for a pre-1.0 crate. Most of the breakage is
deliberate: several APIs that previously returned a value now return a `Result`,
and several paths that previously produced a plausible-looking answer under
violated preconditions now refuse the round instead.

Read this before upgrading under `^0.3`. Nothing here is a silent change.

**Contents**

- [Why the changes are breaking](#why-the-changes-are-breaking)
- [Error types](#error-types)
- [Krum and Multi-Krum](#krum-and-multi-krum)
- [Serialized configuration](#serialized-configuration)
- [Reputation API](#reputation-api)
- [Flower adapter](#flower-adapter)
- [Audit records](#audit-records)
- [Norm-bound filtering](#norm-bound-filtering)
- [Upgrade checklist](#upgrade-checklist)

---

## Why the changes are breaking

Two themes run through the release.

**Refusing instead of guessing.** 0.3.1 accepted non-finite values, proceeded
past Krum's quorum condition with a logged warning, clamped an out-of-range
Multi-Krum `m`, and clamped invalid reputation inputs. Each produced a result
that looked like every other result. 0.4.0 returns a typed error in all four
cases. If your deployment was relying on any of them, it was relying on output
that carried no guarantee.

**Reporting instead of clamping.** Invalid *configuration* -- a NaN reputation
threshold, a negative reward, an unusable norm bound -- is now rejected at the
point it is supplied rather than silently repaired into something usable.

---

## Error types

`QoraError` gained variants and changed one. All are `#[non_exhaustive]`-adjacent
in practice: match with a `_` arm.

### New input-validation variants

| Variant | Raised when |
|---|---|
| `NonFiniteValue { update_index, row, col, value }` | An update contains NaN or infinity. Reported with the exact coordinate. |
| `NonFiniteWeight { index, value }` | A FedAvg sample weight is NaN or infinite. |
| `ClientIdCountMismatch { updates, client_ids }` | `client_ids` length differs from `updates`. Previously **panicked** when too many IDs were supplied, and silently under-attributed reputation when too few. |
| `WeightsNotSupported { method }` | Weights supplied for a method other than FedAvg. Previously ignored silently. |
| `InvalidMultiKrumSelection { clients, byzantine, selected, maximum }` | An explicit Multi-Krum `m` outside `1..=n - 2f - 2`. |
| `NormBoundExceeded { norm, bound }` | A well-formed update exceeds a usable norm bound. `norm` is `f64`. |
| `InvalidNormBound { value }` | A norm bound is not finite and strictly positive. |
| `InvalidAuditEntry { reason }` | An audit entry would not describe a possible aggregation attempt. |

### Reputation validation variants

`InvalidReputationScore`, `InvalidReputationAdjustment`, `InvalidReputationDecay`,
`InvalidReputationThreshold`, and `NonFiniteReputationDistance`. See
[Reputation API](#reputation-api).

### `AllUpdatesRejected` changed shape

**Before (0.3.1 development builds):**

```rust
QoraError::AllUpdatesRejected { total, rejected, threshold }
```

**After:**

```rust
QoraError::AllUpdatesRejected { submitted }
```

Rendered message changed from
`reputation gating rejected all updates: 3 of 3 below threshold 0.5` to
`all 3 submitted updates were rejected`.

**Why.** The variant now covers reputation gating, norm-bound filtering, and any
mixture of the two. A single `threshold` field cannot describe a round where
reputation removed some clients and the norm bound removed the rest, and separate
per-policy variants could not represent the mixed case at all.

**Migration.**

```rust
// Before
Err(QoraError::AllUpdatesRejected { total, rejected, threshold }) => {
    log::warn!("{rejected}/{total} below {threshold}");
}

// After
Err(QoraError::AllUpdatesRejected { submitted }) => {
    log::warn!("all {submitted} updates rejected");
}
```

`total` is the same quantity as `submitted`; `rejected` always equalled it. The
threshold is a value you configured. For per-client reasons, use
[`aggregate_with_audit`](#audit-records).

**Python:** the exception type is unchanged (`ValueError`). Code matching on the
message text needs updating -- match `"were rejected"` rather than
`"reputation gating rejected all"`.

---

## Krum and Multi-Krum

### No best-effort path below quorum

`aggregate_krum` and `aggregate_krum_bfp16` returned `Some(result)` for any
input, printing `WARN: Krum condition not met ... Proceeding with best-effort.`
to stderr when `n < 2f + 3`. That result carried **no Byzantine guarantee** and
was indistinguishable from a sound one.

Both now return `None` below quorum. `ByzantineAggregator` reports
`InsufficientQuorum { needed, actual }`, where `needed` is the calculated
`2f + 3` -- it previously reported a hardcoded `3` regardless of `f`.

```rust
// Find a valid f for the cohort you actually have:
let f = qora_fl::verification::max_tolerable_f(n);
```

### Multi-Krum selection bound enforced

`aggregate_multi_krum_bfp16` clamped `m` with `m.max(1).min(n)`. An oversized
`m` returned a normal-looking selection that admitted Byzantine vectors into the
average. It now returns `None`, and `ByzantineAggregator` reports
`InvalidMultiKrumSelection`.

### `m` is now optional

**Before:** `AggregationMethod::MultiKrum(usize, usize)`
**After:** `AggregationMethod::MultiKrum(usize, Option<usize>)`

```rust
// Before
AggregationMethod::MultiKrum(1, 3)

// After -- explicit: honored exactly, or refused. Never silently reduced.
AggregationMethod::MultiKrum(1, Some(3))

// After -- bare: resolved to min(3, n - 2f - 2) against the accepted cohort.
AggregationMethod::MultiKrum(1, None)
```

The distinction is load-bearing. An omitted `m` is capped to the safe maximum, so
a small cohort selects fewer vectors instead of failing; an explicit `m` is
honored or refused, never rewritten. Python `"multi_krum"` maps to the bare form
and `"multi_krum:f:m"` to the explicit one.

### Preconditions are checked against the accepted cohort

In `n >= 2f + 3` and `1 <= m <= n - 2f - 2`, `n` is the cohort that survives
reputation gating and norm filtering -- not the number submitted. Enabling either
filter can push a previously valid round below quorum. That is reported, never
repaired by reinstating clients or weakening `f`.

---

## Serialized configuration

`ByzantineAggregator` is `Serialize`/`Deserialize` and is persisted through
`to_json`/`from_json` in the Python bindings. Four things changed for stored
state.

### Multi-Krum JSON

`Option<usize>` serializes untagged, so the shape is compatible in both
directions: a 0.3.1 payload `{"MultiKrum":[1,3]}` still deserializes, as
`MultiKrum(1, Some(3))`.

**What changes is meaning, not shape.** A restored 0.3.1 configuration becomes an
**explicit** `m`, so a round where `m > n - 2f - 2` now fails with
`InvalidMultiKrumSelection` instead of silently clamping. That is the intended
effect of the fix, but it surfaces at restore time rather than at upgrade time.

**Action:** either confirm the stored `m` fits your smallest expected cohort, or
rewrite the second element to `null` to adopt the adaptive default.

### The new `norm_bound` field

Carries `#[serde(default)]`, so a 0.3.1 configuration written before the field
existed restores with filtering **disabled**. No action required.

A bound that *is* present is validated on the way in: a stored `0.0`, negative,
or non-finite value fails deserialization rather than silently rejecting every
update at the next round.

### Reputation state is validated on restore

A persisted score outside `[0, 1]`, or a non-finite one, now fails
deserialization. Previously such a file restored into a store whose invariant was
already broken. Same for a persisted `ban_threshold`.

**Action:** if you have stored reputation files from a build that could write
out-of-range scores, clamp or drop the offending entries before upgrading.

### Audit schemas are not wire-compatible

The legacy `verification::audit::AggregationAuditEntry` holds `round`,
`n_clients`, `n_excluded`, a free-form `method` string, and `trim_fraction`. The
new root-level `AggregationAuditEntry` holds a schema version, an
effective-method descriptor, a per-update decision list, and an outcome.

**Stored legacy records cannot be read as the new type. There is no automatic
migration.** Both types still exist; they merely share a name.

---

## Reputation API

Five operations that silently clamped now return `Result`.

| Call | Before | After |
|---|---|---|
| `ReputationStore::set_score` | `()` | `Result<(), QoraError>` |
| `ReputationStore::reward` | `()` | `Result<(), QoraError>` |
| `ReputationStore::penalize` | `()` | `Result<(), QoraError>` |
| `ReputationStore::decay_toward_default` | `()` | `Result<(), QoraError>` |
| `ReputationTracker::decay_toward_default` | `()` | `Result<(), QoraError>` |
| `ByzantineAggregator::decay_reputations` | `()` | `Result<(), QoraError>` |
| `ByzantineAggregator::with_ban_threshold` | `Self` | `Result<Self, QoraError>` |

```rust
// Before
let mut agg = ByzantineAggregator::with_ban_threshold(method, 0.2, 0.3);
agg.decay_reputations(0.02);

// After
let mut agg = ByzantineAggregator::with_ban_threshold(method, 0.2, 0.3)?;
agg.decay_reputations(0.02)?;
```

### Numeric requirements

| Input | Requirement | Rejected with |
|---|---|---|
| Score | Finite, within `[0, 1]` | `InvalidReputationScore` |
| Reward / penalty amount | Finite, non-negative | `InvalidReputationAdjustment` |
| Decay factor | Finite, within `[0, 1]` | `InvalidReputationDecay` |
| Ban threshold | Finite, within `[0, 1]` | `InvalidReputationThreshold` |

Large *valid* amounts still saturate at the boundary; only invalid ones are
refused. A failed operation leaves the previous value intact.

**Why rejection rather than clamping.** The clamps were the vulnerability.
`f32::clamp` propagates NaN, so `set_score(id, NaN)` stored `NaN`. `f32::min`
and `f32::max` *discard* NaN, so `reward(id, NaN)` stored `1.0` and
`penalize(id, NaN)` stored `0.0`. A negative "reward" of `-5.0` drove a score to
`-4.5`, escaping the `[0, 1]` invariant entirely.

`ReputationTracker::reward_valid_zkp`, `penalize_drift`, and
`penalize_zkp_failure` keep their infallible signatures -- their amounts are
crate constants validated at compile time.

In Python these raise `ValueError` through the existing conversion. No new
exception type is introduced.

---

## Flower adapter

### One client result is now one complete update

The adapter previously passed **each layer to the aggregator in a separate
call**. A selection method could therefore choose a different client for each
layer and return a model no client submitted, and reputation was computed from
layer 0 only -- letting a client match the cohort on the first layer and deviate
arbitrarily on every later one without penalty.

Complete models are now flattened into one update, aggregated in a single call,
and split back into their original layer shapes.

**Effect on results:** aggregates will differ from 0.3.1 for multi-layer models
under the selection methods (Krum, Multi-Krum). This is the fix, not a
regression.

### Integer parameter arrays are rejected

Only floating-point dtypes are accepted. `float64` is converted to `float32` and
its extra precision is not preserved. Integer arrays were previously coerced into
trainable floating-point parameters without comment; they now raise `ValueError`
naming the client and layer.

### FedAvg weights by `num_examples`

Previously the adapter passed no weights at all. Robust methods still treat each
accepted client update as one participant -- supplying weights there returns
`WeightsNotSupported` rather than being silently ignored.

### `accept_failures` is honored

Previously inherited from `FedAvg` and ignored. Reported failures now refuse the
round when it is `False`, returning `(None, {})` without invoking Rust, moving
reputation, or calling the metrics callback.

### Automatic metrics were removed

`qora_round`, `qora_num_clients`, `qora_num_filtered`, and the per-client
`reputation_<cid>` entries are **no longer emitted**. They invented an
aggregation policy for arbitrary metric names and grew one entry per client per
round.

```python
# Before: metrics appeared automatically.
# After: supply a callback, or receive {}.
strategy = QoraStrategy(
    aggregation_method="trimmed_mean",
    fit_metrics_aggregation_fn=my_weighted_average,
)
```

Reputation remains available through `QoraStrategy.get_reputation(client_id)`.

### The Python-side reputation filter was removed

The adapter had its own threshold comparison and its own fail-open fallback,
which masked `AllUpdatesRejected` and could apply reputation twice. Gating and
updates now happen once, in the Rust aggregator.

### Flower is still optional

Installed via `pip install "qora-fl[flower]"`, supported range `flwr>=1.5,<2`,
CI-tested at 1.5.0 and 1.32.1. `import qora` works without it; only
`qora.QoraStrategy` requires it and raises an `ImportError` naming the extra.

---

## Audit records

New in 0.4.0, and **opt-in by call site** -- existing code is unaffected.

```rust
let round = agg.aggregate_with_audit(&updates, Some(&client_ids))?;
let aggregate = round.aggregate();
let entry = round.audit();
```

`aggregate_weighted_with_audit` is the weighted counterpart. Both return
`AuditedAggregationError` on failure, which carries the underlying `QoraError`
via `source_error()` and — for an all-rejected round — the record via `audit()`.

- No automatic storage. Qora-FL writes nothing, keeps no log, and retains no
  entry inside the aggregator.
- The legacy `verification::audit` types are **deprecated**, not removed.
- **Not exposed to Python** in 0.4.0.

See [SECURITY.md](../SECURITY.md) for what a record is and is not.

---

## Norm-bound filtering

New in 0.4.0 and **disabled by default**. Existing code that does not configure a
bound computes no norms and behaves exactly as before — there is nothing to
migrate unless you want the feature.

```rust
let agg = ByzantineAggregator::new(method, trim_fraction)
    .with_norm_bound_filter(10.0)?;   // finite and > 0
```

```python
agg = ByzantineAggregator("median", 0.0, norm_bound=10.0)
strategy = QoraStrategy(aggregation_method="median", norm_bound=10.0)
```

Behavior worth knowing before enabling it:

- All-rejected fails closed with the generic `AllUpdatesRejected`.
- A rejected update does **not** receive a reputation penalty.
- Method quorum is evaluated after filtering, so a bound can make Krum or
  Multi-Krum refuse a round. See [docs/TUNING.md](TUNING.md) for choosing a
  bound.

`verification::check_norm_bound` remains public and now returns
`NormBoundExceeded` rather than a free-form `VerificationError`.
`filter_by_norm_bound` is **deprecated** — it discards errors silently and
cannot report per-update reasons. It is not called by the aggregation path.

---

## Upgrade checklist

1. Add `?` or explicit handling to the reputation calls in the table above.
2. Replace `MultiKrum(f, m)` with `MultiKrum(f, Some(m))`, or `None` for the
   adaptive default.
3. Update any `AllUpdatesRejected` destructuring to `{ submitted }`, and any
   Python message matching.
4. Add a `_` arm to `QoraError` matches, or handle the new variants.
5. Confirm your cohort sizes satisfy `n >= 2f + 3`; there is no best-effort path
   below it any more.
6. Check stored `MultiKrum` JSON: an explicit `m` that used to clamp will now
   error.
7. Check stored reputation files for out-of-range scores.
8. If you use the Flower adapter: confirm your parameter arrays are
   floating-point, and supply `fit_metrics_aggregation_fn` if you relied on the
   removed automatic metrics.
9. Expect different aggregates for multi-layer models under Krum and Multi-Krum
   through the Flower adapter.
10. Nothing to do for norm filtering or audit records unless you opt in.
