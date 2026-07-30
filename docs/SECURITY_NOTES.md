# Security Notes

Detailed analysis behind security-relevant changes. Each note records the
behavior that was measured *before* the fix, so the reasoning survives beyond
the changelog entry that summarizes it.

For reporting vulnerabilities, see [SECURITY.md](../SECURITY.md).

---

## 2026-07-29 -- Non-finite input accepted by every aggregation method

Affects all releases up to and including 0.3.1. Fixed on the
`hardening/input-validation` branch.

### Summary

`NaN` and infinity were accepted by every aggregation entry point without
error. All five methods were affected, each failing in a different way, and
three of the five produced finite, plausible-looking output that gave no
indication anything was wrong.

The measurements below come from a probe run against 0.3.1: five updates, four
honest (values near 1.0) and one carrying the non-finite value in every
coordinate.

### Per-method pre-fix behavior

| Method | Input | Pre-fix result |
|---|---|---|
| `trimmed_mean` | NaN | `Ok([1.05, 1.017])` -- finite, order-dependent |
| `median` | NaN | `Ok([1.05, 1.0])` -- finite, silently wrong |
| `fedavg` | NaN | `Ok([NaN, NaN])` |
| `krum` | NaN | `Ok([1.05, 0.95])` -- attacker invisible to the metric |
| `multi_krum` | NaN | `Ok([1.05, 0.95])` -- same |
| all | `inf` / `-inf` | analogous; `fedavg` returned `inf` / `-inf` |

#### `trimmed_mean` and `median`: order-dependent output

Both sort each coordinate with
`values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))`. For `NaN`,
`partial_cmp` returns `None`, so the comparator reports `Equal` against every
other value. `sort_by` is stable, so a `NaN` stays wherever the caller happened
to place it rather than sorting to either end.

The consequence is that the aggregate depended on **which client submitted the
bad value**. With five clients at `trim_fraction = 0.2` (`n_trim = 1`), a single
`NaN` produced:

| NaN submitted by client | Pre-fix result |
|---|---|
| 0 | `1.0` (trimmed away) |
| 1 | `NaN` |
| 2 | `NaN` |
| 3 | `NaN` |
| 4 | `1.0` (trimmed away) |

Since client submission order is not something a server controls, this made the
outcome effectively arbitrary. Regressed by
`trimmed_mean_rejects_nan_at_every_client_position`.

#### `fedavg`: direct propagation

One client's `NaN` reached every coordinate of the output. A non-finite *weight*
did the same thing even when all updates were well-formed -- the case that
motivated `validate_weights`.

#### `krum` / `multi_krum`: the most serious case

`Bfp16Vec::from_f32_slice` launders non-finite values into plausible finite
ones:

```text
from_f32_slice([1.0, NaN, 2.0]) -> exponent -14, mantissas [16384, 0, 32767]
from_f32_slice([1.0, inf, 2.0]) -> exponent 126, mantissas [0, 32767, 0]
```

Two separate mechanisms cause this:

1. The block exponent comes from a `max_abs` reduction built on `f32::max`,
   which **ignores** `NaN` (it returns the non-NaN operand). A `NaN` therefore
   does not influence the chosen exponent.
2. Quantization does `scaled.clamp(-32767.0, 32767.0).round() as i16`. In Rust,
   a float-to-int cast saturates, and `NaN as i16` is **`0`**.

So a `NaN` coordinate is presented to Krum's distance metric as exactly `0.0`.
This is worse than propagation, because it is *advantageous to the attacker*: a
zero contributes nothing to the squared-distance sum, so a `NaN`-poisoned
vector can score as **closer to the honest cluster than a genuine outlier
would**, making it more likely to be selected. And because Krum returns the
original `f32` array rather than the decoded BFP-16 vector
(`agg_updates[best_idx].clone()`), a selected attacker's `NaN` passes through
untouched.

Krum over five all-`NaN` updates returned `Ok([[NaN]])`.

Infinity was *accidentally* contained rather than handled: `inf.log2().ceil()`
saturates the shared exponent to 126, which quantizes every other coordinate in
that vector to zero and pushes the vector past `MAX_BFP16_EXP_DIFF`, so
`dist_sq_bfp16` returns `i64::MAX` and the vector is effectively rejected.
Nothing detected the bad input; it merely quantized into unrecognizability.

Regressed by `aggregate_krum_rejects_all_nan_updates`. The laundering behavior
itself is pinned by `bfp16_encoding_still_launders_non_finite_input` so that a
future change to the constructor is deliberate -- see "Deferred" below.

### Reputation tracking could not see the attacker

`ByzantineAggregator::update_reputations` scores each client on its distance to
the aggregate:

```rust
if distance < 1.0        { reward(id, 0.02) }
else if distance > 10.0  { penalize(id, 0.08) }
```

When a client submits `NaN`, `distance` is `NaN`, and **both** comparisons are
false. Neither branch fires. Measured on 0.3.1 with four honest clients and one
`NaN` client: honest clients reached `0.52` while the `NaN` client stayed at the
`0.5` default.

A `NaN` attacker was therefore never penalized, and consequently never crossed
any ban threshold -- immune to the very mechanism meant to exclude it. This is
why validation must run *before* reputation gating rather than alongside it: the
scoring path cannot be used to enforce a property it cannot observe.

Regressed by `nan_update_does_not_enter_reputation_tracking`.

### Why rejection rather than sanitization

Coercing `NaN` to a finite value is what the BFP-16 encoder already did, and
that is precisely the hole. Any substitution invents data the client did not
send, and for a Byzantine-tolerant aggregator that means inventing data an
*adversary* did not have to send. Refusing the round is the only outcome that
neither fabricates input nor silently corrupts output.

### Deferred

`Bfp16Vec::from_f32_slice` remains public and unchecked, so it is still a
laundering entrance for callers who bypass `ByzantineAggregator`. Closing it is
an API decision rather than a check: either add
`try_from_f32_slice(...) -> Result<Self, QoraError>`, or keep the current
constructor for compatibility while adding a checked constructor and deprecating
the unchecked one.

`aggregate_krum_bfp16` and `aggregate_multi_krum_bfp16` take already-encoded
vectors, so finiteness is not checkable there by construction. The check
necessarily belongs before encoding.

---

## 2026-07-29 -- `client_ids` length was never validated

Affects all releases up to and including 0.3.1.

Supplying **more** client IDs than updates panicked with
`index out of bounds: the len is 3 but the index is 3` inside the ban-threshold
filter, which derives positional indices from `client_ids` and uses them to
index `updates`. The path was reachable through the PyO3 API, so a Python caller
could trigger it.

Supplying **fewer** IDs than updates did not panic and did not error. It
aggregated every update but attributed reputation to only the first
`client_ids.len()` of them, because `update_reputations` zips the two slices and
stops at the shorter one. Clients past the end of the ID list were scored by
nobody -- silently exempt from reputation consequences.

Both directions now return `QoraError::ClientIdCountMismatch { updates,
client_ids }`, checked before any filtering or indexing. Regressed by
`aggregator_rejects_too_many_client_ids` and
`aggregator_rejects_too_few_client_ids`.

---

## 2026-07-29 -- Krum proceeded past its own safety condition

Affects all releases up to and including 0.3.1.

`aggregate_krum` and `compute_krum_scores_bfp16` documented the requirement
`n >= 2f + 3`, then, when it was not met, printed

```text
WARN: Krum condition not met (n=4 < 2*f+3=7). Proceeding with best-effort.
```

to stderr and continued with a clamped neighbor count. The returned value
carried no Byzantine guarantee but was indistinguishable from a sound result --
a warning on stderr is not something a calling service reliably surfaces.

`verification::krum_condition_met(n, f)` already existed and was fully tested,
but was never called from the aggregation path: the check was written and
documented, yet unenforced.

Both low-level paths now call it and return `None`.
`ByzantineAggregator` maps that to
`QoraError::InsufficientQuorum { needed: krum_min_clients(f), actual: n }`. The
`needed` field previously reported a hardcoded `3` for every quorum failure
regardless of `f`, so a configuration needing 7 clients reported that it needed
3.

`krum_min_clients` uses saturating arithmetic, because `f` can reach the library
from untrusted input -- a parsed `"krum:18446744073709551615"` method string
would otherwise overflow `2 * f + 3`.

Regressed by `aggregator_krum_rejects_invalid_condition`,
`aggregator_multi_krum_rejects_invalid_condition`,
`low_level_krum_returns_none_when_condition_is_invalid` (which cross-checks both
paths against `krum_condition_met` over an `n` x `f` grid), and
`krum_min_clients_saturates_on_absurd_f`.
