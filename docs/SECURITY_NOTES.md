# Security Notes

Detailed analysis behind security-relevant changes. Each note records the
behavior that was measured *before* the fix, so the reasoning survives beyond
the changelog entry that summarizes it.

For reporting vulnerabilities, see [SECURITY.md](../SECURITY.md). For the limits
of what 0.4.0's new features do and do not claim, see [Limits of the 0.4.0
additions](#limits-of-the-040-additions) at the end of this file.

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

---

## 2026-07-29 -- Norm-bound verification failed in both directions

Affects all releases up to and including 0.3.1. Fixed on the
`fix/norm-verification` branch, before norm filtering was wired into
aggregation.

### Summary

`check_norm_bound` existed, was tested, and was called by nothing. Its
implementation had two numeric defects and two misattribution defects. Because
no aggregation path used it, none of them could affect a round -- but the
function was public, and a caller implementing its own filtering on top of it
would have inherited all four.

### Measured pre-fix behavior

| Input | Bound | Pre-fix result | Problem |
|---|---|---|---|
| `[1.0, NaN]` | `10.0` | `Err("Update norm NaN exceeds bound 10.0000")` | Malformed input reported as a policy violation, in a message that is not a true statement |
| `[1.0, inf]` | `10.0` | `Err("Update norm inf exceeds bound 10.0000")` | Same |
| `[1e20, 1e20]` (true norm ~1.41e20) | `1e30` | `Err(...)` | **False reject.** `l2_norm_sq` accumulated in `f32`, so the squared sum (1e40) saturated to `inf` and exceeded every finite bound -- for an update four orders of magnitude *inside* its bound |
| `[1e-25, 1e-25]` (true norm ~1.41e-25) | `1e-26` | `Ok(())` | **False accept.** Each squared term (1e-50) underflowed below the smallest `f32` subnormal, so the norm came back exactly `0.0`, which passes any positive bound |
| `[1.0, 0.0]` | `inf` | `Ok(())` | A bound that verifies nothing while appearing configured |
| `[1.0, 0.0]` | `NaN` | `Err(...)` | Rejects everything, with no distinct error |
| `[1.0, 0.0]` | `0.0` / `-1.0` | `Err(...)` | Rejects everything as if oversized |

The underflow row is the one that fails open, and it is the reason this was
fixed before any integration rather than alongside it: a client submitting
sufficiently small coordinates would have passed a bound it genuinely exceeded.

### Fix

Norms accumulate in `f64` via `l2_norm_f64`, which carries `f32::MAX^2`
(~1.2e77) with room to spare. Non-finite coordinates return
`QoraError::NonFiniteValue` with the offending `[row, col]`; an unusable bound
returns `QoraError::InvalidNormBound`; a valid update exceeding a valid bound
returns the typed `QoraError::NormBoundExceeded { norm, bound }` carrying the
measurements as fields. Equality is acceptance.

`filter_by_norm_bound` converted every one of the above into a silent index
drop -- a NaN update simply vanished from the returned indices with no error and
no record. It is deprecated rather than repaired, and the 0.4.0 aggregation path
does not call it.

Regressed by the `src/verification/norm_bound.rs` unit tests, which pin each row
of the table above, and by `tests/norm_bound_filtering.rs`, which pins the same
boundaries through the aggregator.

---

## Limits of the 0.4.0 additions

**Not a fix record.** The two features added in 0.4.0 are recorded here so their
limits sit beside the defect history rather than only in release notes.

### Norm-bound filtering enforces a policy; it does not detect attacks

Disabled by default. When enabled it excludes updates whose `f64` L2 norm
exceeds a caller-chosen bound, and that is the entire claim.

- An accepted update is **not** thereby honest. A poisoned update crafted to
  stay under the bound passes untouched, and staying under a published bound is
  trivial for an attacker who knows it.
- A rejected update is **not** thereby malicious. Legitimate causes include a
  high learning rate, many local epochs, an unusually large local dataset, or a
  bound derived from a different model or training stage.
- The filter has no memory. It makes no cross-round inference and does not move
  reputation in either direction.

Two operational risks:

- **Denial of service by misconfiguration.** A bound below honest variation
  rejects honest clients; if it rejects all of them the round fails closed and
  training stalls. Failing closed is the intended direction -- it is still a
  stall, and it is reachable by setting one number wrong.
- **Cohort reduction below quorum.** Preconditions are checked against the
  accepted cohort, so an attacker who can predict the bound can submit
  deliberately over-bound updates to push `n` below `2f + 3` and force Krum or
  Multi-Krum to refuse rounds it would otherwise have aggregated. A cohort sized
  exactly at `2f + 3` has no margin for a single rejection.

### Audit records are observability, not evidence

`AggregationAuditEntry` is a plain serializable struct. It is not immutable, not
signed, not hashed, not chained, and not ordered across rounds. Qora-FL does not
persist it, so nothing about it is verifiable after the fact beyond what the
caller's own storage guarantees.

Entries carry whatever client identifiers the caller supplied and should be
handled under the caller's privacy policy. They carry no model data and no
hashes of model data -- deliberately, since an unkeyed hash can leak information
while creating a false impression of cryptographic auditability.

A record showing a rejection states what the configured policy did. It is not an
attestation that anything was verified.
