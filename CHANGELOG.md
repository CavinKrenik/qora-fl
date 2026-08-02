# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

Input validation and numeric hardening. Every item below is a
correctness/security fix: in each case the documented contract or the intended
invariant already existed and the implementation contradicted it.

Targeting **0.4.0**, not 0.3.2 — several changes are observable to callers, and
under `^0.3` a patch release would reach them automatically rather than
prompting a review. Version metadata in `Cargo.toml` and
`bindings/python/pyproject.toml` deliberately **remains at `0.3.1`** until the
release is actually cut; the API migrations recorded below take effect in
0.4.0.

### Documentation

No behavior changed in this set; every item aligns a public claim with evidence
that exists in this repository.

- Clarified the assumptions and enforced preconditions for trimmed mean,
  median, Krum, and Multi-Krum. A trim fraction is no longer presented as an
  attacker percentage, and median is stated as requiring *strictly fewer than*
  50% adversarial values per coordinate rather than tolerating 50%.
- Narrowed determinism claims to integer Krum scoring after BFP-16 encoding.
  The previous "deterministic aggregation paths" and "bit-perfect agreement"
  wording covered a pipeline that begins with floating-point encoding and ends
  with `f32` results.
- Documented the Flower adapter's sample weighting, model validation, failure
  policy, metrics behavior, and optional-dependency status, including the
  supported and CI-tested Flower versions.
- Clarified that reputation currently provides tracking and participation
  gating. Cubic influence weighting exists as a utility and is not consumed by
  any aggregation path; it is listed as roadmap work rather than a feature.
- Reframed QRES as project history rather than validation evidence. The
  181-day deployment, ESP32 hardware, and "Slander-Amplification" references
  no longer appear as support for claims about Qora-FL, whose evidence is its
  own tests and experiments.
- Corrected the QRES repository link, which pointed at a 404
  (`CavinKrenik/RaaS` → `CavinKrenik/QRES_RaaS`).
- Added explicit experimental status, a "currently implemented" versus
  "experimental or planned" split, and a validation-status section noting the
  absence of an independent security review.
- Rewrote `SECURITY.md` to separate what the project enforces from what it does
  not provide, and to replace the single project-wide "30% Byzantine" threat
  model with per-method assumptions.
- Removed the published benchmark figures and their directional conclusions.
  The evaluation script and its full configuration remain documented, along
  with the two properties that stop a run from being reproducible evidence: the
  dataset can be substituted silently when the OpenML fetch fails, and each
  configuration runs once, giving no variance estimate. Publishing results again
  requires a pinned dataset, multiple seeds, and saved machine-readable output.
- Documented `verification::norm_bound` and `verification::audit` as
  unintegrated and experimental. Superseded later in this same cycle:
  `norm_bound` is now the shared primitive behind the opt-in aggregation filter,
  and the legacy `verification::audit` types are deprecated in favour of the
  root-level schema. The READMEs, `docs/TUNING.md`, and
  `docs/VERIFICATION_INTEGRATION.md` describe the shipped behavior.
- Fixed a non-working Rust example in the README: `trim_fraction = 0.3` with
  four clients trims every value and returns `InsufficientQuorum`.

### Fixed

- **The Flower adapter aggregated and updated reputation per layer instead of
  once per complete client model.** Each layer was passed to the aggregator in
  a separate call, so a selection method could choose a *different client for
  each layer* and return a model no client submitted. Reputation was computed
  from the first layer only, which let a client match the cohort on layer 0 and
  deviate arbitrarily on every later layer without penalty. Complete models are
  now flattened into one update, aggregated in a single call, and split back
  into their original layer shapes.
- **Flower is now installed in Python CI**, so `QoraStrategy` is imported and
  exercised against real `FitRes`, `Status`, and `Parameters` types. The
  adapter previously had no CI coverage at all -- the job never installed
  Flower, so the import could not even be attempted.

- **Multi-Krum now enforces its documented selection bound
  `1 <= m <= n - 2f - 2`** (Blanchard et al., 2017). `aggregate_multi_krum_bfp16`
  previously clamped with `m.max(1).min(n)`, so an oversized `m` returned a
  normal-looking selection that carried no Byzantine guarantee -- the same
  failure mode as the single-Krum "best-effort" path removed earlier in this
  cycle. The low-level function now returns `None`; `ByzantineAggregator`
  reports the new `QoraError::InvalidMultiKrumSelection`.

  This matters most after reputation gating, where the effective `n` shrinks
  between rounds and a fixed `m` can silently drift out of range.
- Norm-bound verification now computes L2 norms with `f64` accumulation,
  preventing finite large updates from overflowing to infinity and very small
  updates from underflowing to zero.
- `check_norm_bound` now rejects non-finite update coordinates with
  `QoraError::NonFiniteValue`.
- Zero, negative, NaN, and infinite norm bounds now return
  `QoraError::InvalidNormBound`.
- Norm-bound rejection messages now report the finite computed norm using
  scientific notation.

### Deprecated

- Deprecated `verification::audit::AggregationAuditEntry` in favor of the new
  root-level `AggregationAuditEntry` schema. The two are different types that
  happen to share a name, and **their serialized shapes are not compatible**:
  the legacy record holds `round`, `n_clients`, `n_excluded`, a free-form
  `method` string, and `trim_fraction`, while the new one holds a schema
  version, the full `AggregationMethod`, a per-update decision list, and an
  outcome. Stored legacy records cannot be read as the new type; there is no
  automatic migration.
- Deprecated `verification::audit::AuditLog`. Audit persistence is now
  caller-owned and **no replacement storage implementation is provided** --
  serialize `AggregationAuditEntry` and store the records with whatever the
  application already uses.

  Neither type is removed, so 0.3 callers keep compiling; both will be removed
  in a future breaking release. The `verification::` re-export carries a
  statement-scoped `#[allow(deprecated)]` so the re-export itself does not warn
  while callers reaching the types through it still do.
- `filter_by_norm_bound` is deprecated because it silently discards
  verification errors. Callers should use `check_norm_bound` and handle each
  result explicitly.

### Security

- **Non-finite values in client updates are now rejected** instead of silently
  corrupting the aggregate. New `QoraError::NonFiniteValue { update_index, row,
  col, value }`, enforced by `validation::validate_updates` and
  called from `ByzantineAggregator::aggregate` as well as from `trimmed_mean`,
  `median`, and `fedavg` directly, so the guarantee does not depend on which
  entry point a caller uses.

  All five methods previously accepted non-finite input without error, each
  failing differently: the coordinate-wise methods became order-dependent (a
  NaN's effect varied with which client submitted it), `fedavg` propagated it to
  every output coordinate, and `krum` / `multi_krum` could not observe it at all
  because BFP-16 encoding quantizes NaN to a zero mantissa -- which could make a
  poisoned vector score as *closer* to the honest cluster than a genuine
  outlier. Measured pre-fix behavior for each method is recorded in
  [docs/SECURITY_NOTES.md](docs/SECURITY_NOTES.md).

- **Non-finite client weights are now rejected.** New
  `QoraError::NonFiniteWeight { index, value }`. Updates could be entirely
  well-formed while a single NaN weight destroyed the weighted mean.

- **Non-finite updates were invisible to reputation tracking.** A NaN
  distance-to-aggregate satisfies neither `distance < 1.0` nor
  `distance > 10.0`, so the offending client was never rewarded *or* penalized,
  leaving it permanently unbannable. Validation now runs before reputation
  gating, so such a round is refused outright and no score moves.

- **`client_ids` / `updates` count mismatches now return an error instead of
  potentially panicking.** New `QoraError::ClientIdCountMismatch { updates,
  client_ids }`, checked before any filtering or indexing. Supplying more IDs
  than updates previously panicked with `index out of bounds` inside the
  ban-threshold filter, a path reachable through the PyO3 API. Supplying fewer
  silently succeeded while attributing reputation to only a prefix of the
  clients, because `update_reputations` zips and stops at the shorter side.

### Changed

- **The Flower strategy now validates complete client model structures**,
  aggregates each client model as one flattened update, and restores the
  original layer shapes afterward. Every client is checked against the first
  as structural reference: layer count, per-layer shape, floating-point dtype,
  and finiteness. A model with the same total element count but a different
  layer structure -- `[(10, 4), (4,)]` versus `[(44,)]` -- is refused rather
  than reconciled by broadcasting.

  A malformed *successful* result raises `ValueError` rather than being
  dropped. Silently discarding it would change the round's effective
  adversarial fraction, which is exactly the quantity Krum's `n >= 2f + 3`
  condition is stated over. Messages name the client and layer index.

  **Migration:** **integer parameter arrays are now rejected.** Only
  floating-point dtypes are accepted; `float64` is converted to `float32` and
  its extra precision is not preserved. Integer arrays were previously coerced
  into trainable floating-point parameters without comment. A deployment
  sending integer layers will now see a `ValueError` naming the client and
  layer rather than silently different aggregation.

- **Flower FedAvg now weights updates by each result's `num_examples`.** New
  `ByzantineAggregator::aggregate_weighted` carries the weights to the Rust
  `fedavg`, which already accepted them but was always called with `None`.
  Weights are filtered alongside updates when reputation gating removes a
  client, so they cannot be applied to the wrong participant.

- **Robust methods continue to treat accepted client updates equally.**
  `num_examples` weighting applies to FedAvg only; supplying weights with any
  other method returns the new `QoraError::WeightsNotSupported` rather than
  being silently ignored. Reweighting a median or trimmed mean by claimed
  sample count would hand an attacker proportional influence simply for
  claiming a large dataset -- a change to the threat model, not a detail.

- **The strategy now honors `accept_failures`.** Reported failures refuse the
  round when it is False, returning `(None, {})` without invoking Rust,
  moving reputation, or calling the metrics callback. Previously the parameter
  was inherited from `FedAvg` and ignored.

- **Reputation gating and updates are performed once, by the Rust aggregator,
  using each client's complete model update.** The adapter's second,
  Python-side reputation filter is removed. It used its own threshold
  comparison and its own fail-open fallback, which masked
  `AllUpdatesRejected` and could apply reputation twice.

- **Metrics are aggregated only through `fit_metrics_aggregation_fn`.** When
  none is configured the strategy returns `{}`. The previous `qora_round`,
  `qora_num_clients`, `qora_num_filtered`, and per-client `reputation_<cid>`
  entries are no longer emitted: they invented an aggregation policy for
  arbitrary metric names and grew one entry per client per round. Reputation
  remains available through `QoraStrategy.get_reputation`.

- Python `ByzantineAggregator.aggregate` accepts an optional `weights`
  argument. Existing two-argument calls are unaffected.

- **Flower remains an optional dependency, installed via `qora-fl[flower]`.**
  `import qora` and the whole core API -- `ByzantineAggregator`,
  `ReputationManager` -- work with Flower absent; only `qora.QoraStrategy`
  requires it, and raises an `ImportError` naming the extra when it is
  missing. Verified by installing the built wheel without the extra into a
  clean environment.

  The extra is now bounded: **supported range `flwr>=1.5,<2`**, upper-bounded
  because the adapter targets Flower's 1.x strategy interface. **Tested in CI:
  1.5.0 and 1.32.1** -- the workflow runs a matrix over both ends of the range
  rather than installing whichever version a resolver picks, which would be no
  evidence for the range at all. Versions in between are covered by the
  declared promise but are not individually exercised.

- The adapter is described as a "Flower-compatible strategy adapter" rather
  than a "drop-in" replacement, in both READMEs. It does not reproduce every
  behavior of Flower's built-in `FedAvg`.

- **Reputation state now maintains a numeric invariant: every stored score is
  finite and within `[0, 1]`, and every operation preserves it.** Each mutation
  path previously had a way to break it, and the clamps intended to prevent
  that were themselves the vulnerability:

  | Call | Before | Now |
  |---|---|---|
  | `set_score(id, NaN)` | stored `NaN` (`f32::clamp` propagates it) | `InvalidReputationScore` |
  | `reward(id, NaN)` | stored `1.0` (`f32::min` *discards* NaN) | `InvalidReputationAdjustment` |
  | `penalize(id, NaN)` | stored `0.0` (`f32::max` discards NaN) | `InvalidReputationAdjustment` |
  | `reward(id, -5.0)` | stored `-4.5` | `InvalidReputationAdjustment` |
  | `penalize(id, -5.0)` | stored `5.5` | `InvalidReputationAdjustment` |
  | `decay_toward_default(NaN)` | turned **every** score into `NaN` | `InvalidReputationDecay`, nothing mutated |

  The NaN reward and penalty cases are the dangerous ones: they produced no
  NaN at all, just a plausible extreme -- silently fully-trusted or silently
  banned, with nothing for an operator to notice.

  Invalid operations are atomic: the targeted score is unchanged and no other
  entry is touched. The decay factor in particular is validated *before* the
  mutation loop, so a rejected call cannot leave the store partially decayed.

  Out-of-range values are **rejected, not clamped**. A caller passing `5.0`
  has misread the scale, and silently storing `1.0` would conceal that.

  The accepted ranges, in full:

  | Input | Must be | Rejected with |
  |---|---|---|
  | Stored score | finite, `0.0 <= s <= 1.0` | `InvalidReputationScore` |
  | Reward / penalty amount | finite, `>= 0.0` (any magnitude) | `InvalidReputationAdjustment` |
  | Decay factor | finite, `0.0 <= f <= 1.0` | `InvalidReputationDecay` |
  | Ban threshold | finite, `0.0 <= t <= 1.0` | `InvalidReputationThreshold` |

  Only amounts are unbounded above, because the resulting score is clamped
  after the arithmetic: `reward(id, 10.0)` still yields `1.0`, and
  `penalize(id, 20.0)` still yields `0.0`. A decay factor of `0.0` is a no-op
  and `1.0` restores the default.

- **Reputation distance is computed in `f64`, with operands widened before
  subtraction.** The previous `diff.iter().map(|x| x * x).sum::<f32>().sqrt()`
  overflowed to infinity for large finite differences and underflowed to zero
  for tiny ones. Casting the final sum would not have been enough: the `f32`
  subtraction itself overflows before any wider arithmetic sees the values.
  A non-finite distance now returns `QoraError::NonFiniteReputationDistance`
  rather than being read as zero distance, which would mean maximum trust.

- **Ban thresholds are validated.** `ByzantineAggregator::with_ban_threshold`
  returns `Result` and rejects a non-finite or out-of-range threshold. This
  closes a fail-open: gating activates on `ban_threshold > 0.0`, which is false
  for `NaN`, so a non-finite threshold disabled the gate entirely while
  appearing configured. Thresholds above the 0.5 default score remain valid --
  they deliberately reject unknown clients.

- **Deserialization enforces the same rules.** `ReputationStore` rejects a
  payload containing a score outside `[0, 1]`, and a persisted
  `ByzantineAggregator` with an invalid `ban_threshold` is refused. Without
  this the guarantees would hold only for callers who never restore state.
  Corrupted data is refused rather than silently clamped.

- **Gating treats non-finite scores as untrusted.** `is_banned` and
  `count_below` now count a non-finite score as below any threshold.
  Enforcement should make this unreachable; it is defence in depth for state
  written by an older version. Without it, `NaN < threshold` is false, so a
  poisoned client was permanently *unbannable* and invisible to adaptive
  trimming.

  **Migration (0.4).** These mutation methods now return
  `Result<(), QoraError>` instead of `()`, and `with_ban_threshold` returns
  `Result<Self, QoraError>`:

  | Method | Was | Now |
  |---|---|---|
  | `ReputationStore::{set_score, reward, penalize, decay_toward_default}` | `()` | `Result<(), QoraError>` |
  | `ReputationTracker::decay_toward_default` | `()` | `Result<(), QoraError>` |
  | `ByzantineAggregator::decay_reputations` | `()` | `Result<(), QoraError>` |
  | `ByzantineAggregator::with_ban_threshold` | `Self` | `Result<Self, QoraError>` |

  Rust callers must handle or propagate the result. `ReputationTracker`'s
  `reward_valid_zkp`, `penalize_drift`, and `penalize_zkp_failure` keep their
  infallible signatures: their amounts are crate constants validated at compile
  time, so there is no runtime failure to report. In Python the same operations
  raise `ValueError` through the existing error conversion; no new exception
  type is introduced.

- **`QoraError::AllUpdatesRejected` is now generic about which policy rejected
  the round.** It was `{ total, rejected, threshold }`, shaped for reputation
  gating; it is now `{ submitted }`.

  Norm-bound filtering raises the same variant, and a round can mix the two --
  reputation removing some clients and the bound removing the rest -- so a
  single threshold field cannot describe the outcome truthfully, and separate
  per-policy variants could not represent a mixed rejection at all. Detailed
  per-update reasons belong in `AggregationAuditEntry`, which
  `aggregate_with_audit` returns alongside this error.

  **Migration.** Rust callers destructuring the variant must replace
  `{ total, rejected, threshold }` with `{ submitted }`; `total` is the same
  quantity under its new name, and `rejected` always equalled it. The rendered
  message changed from `reputation gating rejected all updates: 3 of 3 below
  threshold 0.5` to `all 3 submitted updates were rejected`, so Python code
  matching on the message text needs updating -- the exception type is unchanged
  (`ValueError`). Both fields dropped were derivable or constant; nothing that
  was recoverable from the error is now unavailable except the threshold, which
  the caller configured and the audit record reports per client.

- **Method quorum and Multi-Krum selection constraints are evaluated against the
  final accepted cohort.** They already were for reputation gating; norm
  filtering makes the distinction load-bearing, because a bound can shrink a
  cohort that satisfied `n >= 2f + 3` into one that does not, or push an
  explicit Multi-Krum `m` above `n - 2f - 2`. Both return their existing typed
  errors (`InsufficientQuorum`, `InvalidMultiKrumSelection`) reporting the
  accepted count. Filtering is never undone to satisfy a method: no client is
  reinstated, `f` is not weakened, an explicit `m` is not reduced, and no
  best-effort result is returned.

- **`check_norm_bound` returns `QoraError::NormBoundExceeded` instead of
  `QoraError::VerificationError`** for a valid update exceeding a valid bound.
  The measurements are structural fields rather than text inside a message, so
  the aggregation path records them without parsing. `NonFiniteValue` and
  `InvalidNormBound` are unchanged.

- **All aggregation entry points now share one internal engine.** `aggregate`,
  `aggregate_weighted`, and the two audited methods delegate to it, so
  validation, filtering, effective-parameter resolution, and reputation updates
  each run exactly once and the audited and ordinary APIs cannot disagree. Each
  candidate carries its update, client ID, sample weight, and original index as
  one record, so a filter removes them together rather than maintaining parallel
  vectors -- the alignment failure class already fixed once in the Flower
  adapter.

- **Reputation participation gating now fails closed.** When every submitted
  client is below the configured `ban_threshold`, aggregation returns
  `QoraError::AllUpdatesRejected` instead of silently restoring the banned
  clients and aggregating them anyway.

  The previous fallback defeated the gate in the one round where it mattered
  most -- the round where reputation distrusted the entire cohort. A client the
  configured policy rejected is no longer reinstated merely because no other
  client qualified either.

  The error path performs no aggregation and moves no reputation scores, so a
  caller retrying a rejected round does not compound penalties. Gating still
  applies only when `client_ids` are supplied and `ban_threshold > 0`, so
  callers using the default constructor, or aggregating without client IDs,
  are unaffected. Note that unknown clients start at 0.5: a threshold above
  that now rejects a first-round cohort outright rather than passing it
  through ungated.

- **Krum and Multi-Krum now refuse inputs violating `n >= 2f + 3`.**
  `aggregate_krum` and `aggregate_krum_bfp16` return `None`;
  `ByzantineAggregator` reports `QoraError::InsufficientQuorum` with the
  **calculated** requirement via the new
  `verification::krum_min_clients(f)`.

  Both previously logged `WARN: Krum condition not met ... Proceeding with
  best-effort.` to stderr and returned a result that carried no Byzantine
  guarantee while being indistinguishable from a sound one. The `n >= 2f + 3`
  requirement was already documented on these functions; the best-effort path
  contradicted it.

  The `Some` -> `None` transition is observable, but it is classified here as a
  correctness/security fix rather than a breaking change: the mathematical
  requirement predates it and the old behavior violated the documented
  contract. Callers who need a valid `f` for a given `n` can use
  `verification::max_tolerable_f(n)`.

- `ByzantineAggregator` previously reported `InsufficientQuorum { needed: 3 }`
  for *every* Krum quorum failure regardless of `f`. It now reports the real
  figure (`2f + 3`) -- e.g. `needed: 7` for `f=2`, where it used to say `3`.
- `krum_condition_met` now delegates to `krum_min_clients`, so `2 * f + 3`
  saturates instead of overflowing for an absurd `f` reaching the library from
  untrusted input (such as a parsed `"krum:18446744073709551615"` method
  string).
- Krum's neighbor-count clamp `if n > f + 2 { n - f - 2 } else { 1 }` is now
  plain `n - f - 2`; enforcing the quorum condition guarantees
  `k >= f + 1 >= 1`, making the fallback unreachable.
- In `trimmed_mean`, update validation now runs before the `trim_fraction`
  range check. A call with both a dimension mismatch and an out-of-range
  fraction now reports `DimensionMismatch` rather than `InvalidTrimFraction`.
- **`AggregationMethod::MultiKrum(usize, usize)` is now
  `MultiKrum(usize, Option<usize>)`.** The `Option` lets the aggregation
  boundary distinguish an omitted `m` from an explicitly requested one:
  `None` caps to `min(3, n - 2f - 2)`, while `Some(m)` is honored exactly or
  refused. Without the distinction, a bare `"multi_krum"` could not both
  preserve the historical `m = 3` for large cohorts and stay inside the bound
  for 5- and 6-client rounds.

  **Migration.** `Option<usize>` serializes untagged, so the JSON is
  compatible in both directions: a 0.3.1 payload `{"MultiKrum":[1,3]}` still
  deserializes under 0.4.0, as `MultiKrum(1, Some(3))`, and `Some(3)` still
  serializes to `[1,3]`. The omitted form is the new `{"MultiKrum":[1,null]}`.

  What changes is *meaning*, not shape. A persisted 0.3.1 configuration
  restores as an **explicit** `m`, so a round where `m > n - 2f - 2` now fails
  with `InvalidMultiKrumSelection` instead of silently clamping. That is the
  intended effect of the fix, but it can surface at restore time rather than at
  upgrade time. Operators restoring 0.3.1 state should either confirm the
  stored `m` fits their smallest expected cohort, or rewrite the second element
  to `null` to adopt the adaptive default. Rust callers constructing the enum
  directly must update `MultiKrum(f, m)` to `MultiKrum(f, Some(m))`.
- Python `"multi_krum"` (no parameters) now selects `min(3, n - 2f - 2)`
  vectors instead of a fixed 3, so 5- and 6-client rounds are safe rather than
  silently outside the guarantee. Explicit `"multi_krum:f:m"` is unchanged
  where valid and rejected where not; it is never silently rewritten.
- `l2_norm` now uses an internal `f64` accumulator before returning `f32`,
  improving accuracy at extreme finite magnitudes.
- `l2_norm_sq` is documented as able to overflow to infinity or underflow to
  zero when the true squared norm falls outside `f32` range. This is a
  limitation of its return type rather than of its accumulation, so it cannot
  be removed without changing the signature; verification uses `l2_norm`
  instead.

### Added

- **Optional norm-bound filtering on `ByzantineAggregator`.** Filtering is
  **disabled by default** and excludes updates whose finite `f64` L2 norm
  exceeds a configured positive bound.

  Configured with `with_norm_bound_filter(bound) -> Result<Self, QoraError>`,
  cleared with `without_norm_bound_filter()`, read back with `norm_bound()`. In
  Python: `ByzantineAggregator(method, trim_fraction, norm_bound=...)` and
  `QoraStrategy(..., norm_bound=...)`, where `None` is disabled. A bound must be
  finite and strictly positive; zero, negative, NaN, and infinite bounds return
  `QoraError::InvalidNormBound` (Python `ValueError`) rather than being clamped,
  and the same validation runs on deserialization.

  With no bound configured, no norm is computed and behavior is exactly as
  before. With one configured:

  - The norm is computed in `f64` and compared inclusively -- a norm exactly
    equal to the bound participates, matching `check_norm_bound`, which now
    shares the comparison through the internal `evaluate_norm_bound`.
  - A rejected update's client ID and FedAvg sample weight are removed with it,
    so a rejected weight cannot reach the numerator or the denominator or attach
    to another client.
  - **A rejected update does not affect reputation** -- no reward, no penalty,
    no ban, and its stored score does not move however many rounds it is
    excluded for. A norm bound is a participation filter, not a reputation
    policy; a large norm is a policy violation rather than proof of malicious
    intent, and this filter does not claim to detect Byzantine behavior. (This
    reverses the tentative position in
    `docs/VERIFICATION_INTEGRATION.md` §7, which is amended in place.)
  - Reputation gating runs first, and a client it removes is not measured again,
    so every submitted update carries at most one rejection reason.
  - Malformed input still outranks filtering: empty batches, dimension
    mismatches, non-finite values or weights, and client-ID count mismatches are
    errors, never filtering decisions.
  - If nothing survives, the round fails closed with
    `QoraError::AllUpdatesRejected`.

  `verification::filter_by_norm_bound` remains deprecated and is **not** used by
  the integration: it discards errors silently and cannot report per-update
  reasons.

- **Audited aggregation.** `ByzantineAggregator::aggregate_with_audit` and
  `aggregate_weighted_with_audit` return an `AuditedAggregation` -- the
  aggregate plus the caller-owned `AggregationAuditEntry` schema added earlier
  in this cycle -- or an `AuditedAggregationError`.

  The error type exists because `Result<_, QoraError>` would discard the record
  in exactly the round it matters most: `AuditedAggregationError::audit()`
  carries the per-update reasons through an all-rejected failure. It exposes
  `source_error()`, `audit()`, and `into_parts()`, and converts into
  `QoraError`.

  Records are handed to the caller and not retained. Nothing is persisted or
  logged, no entry is stored inside the aggregator, and the serialized
  aggregator does not grow with the round count.

  Which failures carry a record: an all-rejected round does; input validation,
  invalid configuration, a method-precondition failure after some candidates
  survived, and a post-aggregation reputation failure do not. Schema version 1
  has two outcomes -- aggregated, and all updates rejected -- and a round where
  some candidates survived but Krum refused the reduced cohort is neither.
  Rather than attach a record describing something that did not happen, the
  typed error is returned alone. A future schema version may add a failure
  outcome.

  A record from an all-rejected round reports **no effective method
  parameters**: nothing executed, so `effective_trim_fraction` and `effective_m`
  are `None` rather than values that were never applied.

- `QoraError::NormBoundExceeded { norm, bound }` -- a well-formed update
  violating a usable bound. Structural, so the audit integration records the
  measurements without parsing a rendered message; `norm` is `f64`, matching the
  computation. Distinct from `NonFiniteValue` (malformed update, no meaningful
  norm) and `InvalidNormBound` (unusable threshold).
- `tests/norm_bound_filtering.rs` (30 tests) and `tests/audited_aggregation.rs`
  (24 tests): default-disabled behavior, bound validation at construction and on
  deserialization, 0.3.1 configurations restoring with filtering off, boundary
  behavior at and either side of the bound, `f64` exactness at 1e20 and 1e-25,
  weight and client-ID alignment through a rejection, reputation left untouched,
  fail-closed for reputation-only, norm-only, and mixed rejection, method quorum
  evaluated against the accepted cohort, and the audit record's contents,
  effective parameters present after a successful round and absent after a
  refused one, and JSON round trip. 22 Python tests cover the exposed
  configuration through the bindings and the Flower adapter.
- `src/validation.rs`: `validate_updates`, `validate_weights`,
  `validate_client_ids`, plus module documentation recording why non-finite
  input is rejected rather than sanitized. Crate-private, and located at the
  crate root rather than under `aggregators` because `verification` needs it
  too and `aggregators` already depends on `verification`.
- `verification::krum_min_clients(f)` -- minimum client count for a given `f`,
  with saturating arithmetic. Re-exported from `verification`.
- `verification::max_multi_krum_m(n, f)` -- largest `m` Multi-Krum may select
  for a given `(n, f)`, or 0 when the quorum condition fails. Re-exported from
  `verification`.
- `QoraError::InvalidMultiKrumSelection { clients, byzantine, selected,
  maximum }` -- raised for an explicit `m` outside the safe range.
- **Experimental, versioned `AggregationAuditEntry` schema** (`src/audit.rs`)
  for recording per-update acceptance and rejection decisions. Audit records
  are caller-owned: Qora-FL does not persist them, does not retain them inside
  `ByzantineAggregator`, and the aggregation API does not yet return them.
  Integration is future work.

  One `AggregationAuditDecision` per submitted update, carrying its original
  index, an optional client ID, and a typed `AggregationDecision` -- so every
  input has exactly one disposition and the counts are derived rather than
  stored alongside. Rejection reasons are typed
  (`AggregationRejectionReason::{ReputationBelowThreshold, NormBoundExceeded}`)
  rather than strings; the norm is `f64`, matching `check_norm_bound`.
  `AggregationAuditOutcome` distinguishes `Aggregated` from
  `AllUpdatesRejected`, so a round that produced nothing can still explain why.

  `method` records the **effective** parameters via the new
  `AuditedAggregationMethod`, not the configuration-only `AggregationMethod`.
  Two methods resolve parameters at aggregation time, and a record of the
  configuration alone cannot explain what ran: under `adaptive_trim` the trim
  fraction is recomputed each round, and `AggregationMethod::TrimmedMean`
  carries no fraction at all; a bare `"multi_krum"` resolves `m` to
  `min(3, n - 2f - 2)`. The descriptor keeps both sides -- `configured` and
  `effective` trim fractions with an `adaptive` flag, and `requested_m`
  alongside `effective_m` -- so a reader can tell a deliberate setting from a
  runtime resolution. Being an enum, it also makes nonsense combinations such
  as a FedAvg record carrying a trim fraction unrepresentable.

  **The runtime-resolved values are `Option`.** `effective_trim_fraction` is
  `Option<f32>` and `effective_m` is `Option<usize>`, `Some` exactly when the
  outcome is `Aggregated`. A round that rejected every update resolved nothing:
  no fraction was applied and no vectors were selected, so recording a value
  there would name a parameter that never executed. This matters most for a
  bare `"multi_krum"`, whose `m` is resolved against the accepted cohort --
  with an empty cohort there is nothing to cap the default against.
  Configuration is still reported in full, since the configured fraction, the
  `adaptive` flag, `f`, and `requested_m` are known regardless of what ran.

  Method parameters are validated too, on construction *and* deserialization:
  trim fractions must be finite and within `0.0..=0.5`, a non-adaptive record
  must apply its configured fraction, `2f + 3` must not overflow, `effective_m`
  must be at least 1, and an explicit `requested_m` must equal `effective_m` --
  an explicit request is honored exactly or refused, never silently resolved to
  something else. The effective values are checked against the outcome in both
  directions: `Aggregated` carrying `None` is rejected because something ran and
  the parameter it ran with exists, and `AllUpdatesRejected` carrying `Some` is
  rejected because nothing ran. `effective_m` is deliberately not checked
  against the cohort size, which the entry does not know.

  Entries carry no timestamps, round numbers, experiment identifiers, or model
  data -- the library has no clock, no notion of a training round, and no
  reason to copy update values into a record. Callers wrap the entry in their
  own structure when they need those.

  Invariants (one decision per update, contiguous indices, outcome consistent
  with the accepted count, finite in-range measurements) are enforced on
  construction *and* deserialization, so an entry cannot describe an
  impossible attempt. `AGGREGATION_AUDIT_SCHEMA_VERSION` versions the
  serialized shape, not the package; the rejection-reason and outcome enums are
  `#[non_exhaustive]`. Wire stability is not promised while the schema is
  experimental, and a serializable record is not a tamper-proof audit log.

  The pre-existing `verification::audit` types are unchanged and remain unused;
  they are documented as superseded.
- `QoraError::InvalidAuditEntry { reason }` -- raised when an audit entry would
  not describe a possible aggregation attempt.
- `QoraError::AllUpdatesRejected { submitted }` -- raised when participation
  filtering leaves no client.
- `QoraError::InvalidReputationScore`, `InvalidReputationAdjustment`,
  `InvalidReputationDecay`, `InvalidReputationThreshold`, and
  `NonFiniteReputationDistance` -- typed rejections for each reputation input
  class, so a caller can tell an invalid score from an invalid amount, decay
  factor, or threshold.
- `src/reputation/validate.rs`: the four numeric rules in one place, with the
  reasoning for rejecting rather than clamping.
- `tests/reputation_numeric.rs`: 27 integration tests covering each rejection,
  atomicity of failed operations, saturation of large valid amounts,
  deserialization rejection, and end-to-end aggregation keeping every score in
  range. Plus unit tests for the read-side defences and for the `f64` distance
  at both extremes, and 11 Python tests over the exposed methods.
- `tests/reputation_gating.rs`: 12 integration tests covering the typed error
  and its fields, non-restoration of a partially rejected cohort, precedence
  over method execution, reputation state being unchanged on the error path,
  and validation still outranking gating. Five Python tests cover the same
  behavior through the bindings.
- `tests/multi_krum_bounds.rs`: 13 integration tests covering the cap applied
  to an omitted `m`, rejection of explicit out-of-range values, quorum
  precedence, low-level `None` returns, and serde round-trips of both forms.
  Six Python tests cover the same contract through the bindings.
- `docs/SECURITY_NOTES.md` recording the measured pre-fix behavior behind each
  fix in this release.
- 33 new tests, bringing the suite from 145 to 178:
  - 20 integration tests in the new `tests/input_validation.rs`, each
    documenting the pre-fix behavior it pins.
  - 13 unit tests: 12 in `src/aggregators/validate.rs`, 1 in
    `src/verification/krum_condition.rs`.
  - Totals by target: 103 unit + 74 integration (54 `aggregator_tests` + 20
    `input_validation`) + 1 doctest.

### Known limitations (deliberately deferred)

- **`Bfp16Vec::from_f32_slice` does not validate its input.** It remains a
  laundering entrance for callers who bypass `ByzantineAggregator`: `NaN`
  becomes a zero mantissa, and `inf` saturates the shared exponent to 126,
  which quantizes every other coordinate in that vector to zero. The behavior
  is documented on the method and pinned by the
  `bfp16_encoding_still_launders_non_finite_input` test so that changing it is
  a deliberate decision.

  Closing this requires an API decision, not just a check -- either add
  `Bfp16Vec::try_from_f32_slice(...) -> Result<Self, QoraError>`, or keep the
  current constructor for compatibility while adding a checked constructor and
  deprecating the unchecked one. Tracked as a separate roadmap item.

- `aggregate_krum_bfp16` / `aggregate_multi_krum_bfp16` accept already-encoded
  vectors, so finiteness is not checkable there by construction. The check
  necessarily belongs before encoding, which is where
  `ByzantineAggregator::aggregate` performs it.

- **Audit records cover two outcomes, not every failure.** Schema version 1
  distinguishes a round that aggregated from one where filtering rejected
  everything. A method-precondition failure after some candidates survived, and
  a reputation failure after aggregating, are neither, so no record is produced
  for them -- `AuditedAggregationError::audit()` returns `None` and the typed
  error is returned alone. Representing them requires a new outcome and a schema
  version bump.

- **The norm bound is a fixed value the caller picks.** Adaptive or
  percentile-derived bounds are a separate design and deliberately not the first
  version: an adaptive bound is attacker-influenceable, since enough colluding
  clients can drag it upward.

- **Audit records are not exposed to Python.** The Rust API returns typed,
  versioned entries; mirroring that as a hierarchy of Python classes would
  freeze a binding design while the schema is still experimental. A later
  binding can return the serialized shape as a dictionary. Norm-bound
  *configuration* is exposed.

## [0.3.1] - 2026-02-08

### Added
- **Multi-Krum aggregation**: Select top-m updates by Krum score and average them
  - `AggregationMethod::MultiKrum(f, m)` in Rust
  - Python: `"multi_krum"` (defaults f=1, m=3) or `"multi_krum:f:m"`
  - Smoother convergence than single Krum while maintaining Byzantine robustness
- **Generic `ReputationStore<ID>`**: Unified reputation system (`src/reputation/store.rs`)
  - Shared by both `ReputationTracker` (byte-array IDs) and `ByzantineAggregator` (string IDs)
  - Methods: `reward`, `penalize`, `is_banned`, `influence_weight`, `decay_toward_default`, `prune_near_default`
  - Backward-compatible serialization via `#[serde(transparent)]`
- **Adaptive trim fraction**: Dynamic `trim_fraction` from reputation distribution
  - `ByzantineAggregator::set_adaptive_trim(true)` / `adaptive_trim=True` in Python
  - Automatically increases trimming when many clients have low reputation
- **Verification module** (`src/verification/`):
  - `check_norm_bound` / `filter_by_norm_bound`: L2 norm verification for updates
  - `krum_condition_met` / `max_tolerable_f`: Krum safety condition checks
  - `AuditLog`: Append-only aggregation audit log for post-hoc analysis
- **Math module** (`src/math/`): `l2_norm` and `l2_norm_sq` utilities
- **Python API additions**:
  - `ReputationManager.decay(rate)` and `ReputationManager.influence_weight(client_id)`
  - `ByzantineAggregator` accepts `adaptive_trim` parameter
- Multi-Krum Criterion benchmark (`multi_krum_m3`)
- 39 new tests (142 total: 90 unit + 51 integration + 1 doctest)

### Changed
- Krum score computation extracted to shared `compute_krum_scores_bfp16` helper (used by both Krum and Multi-Krum)
- `ByzantineAggregator` reputation field changed from `HashMap<String, f32>` to `ReputationStore<String>`
- `ReputationTracker` now wraps `ReputationStore<[u8;32]>` via `#[serde(flatten)]`
- `PyReputationManager` refactored to wrap `ReputationStore<String>`

## [0.3.0] - 2026-02-06

### Added
- **Flower integration**: `QoraStrategy` drop-in replacement for FedAvg
  - Byzantine-tolerant aggregation in Flower with one line change
  - Reputation-based client filtering per round
  - Optional dependency: `pip install qora-fl[flower]`
- **ReputationManager**: Standalone Python class for persistent trust scores
  - String-based client IDs, reward/penalize API
  - JSON serialization for persistence between server restarts
- **Serialization**: `ByzantineAggregator.to_json()` / `from_json()` for state persistence
- **client_ids parameter**: `aggregate()` now accepts optional client identifiers for reputation tracking
- **Benchmarks**:
  - `examples/benchmark_overhead.py` - Aggregation overhead measurement
  - `examples/mnist_poisoning_demo.py` - FedAvg vs Qora-FL under 30% label-flipping attack
  - `benches/aggregation.rs` - Criterion benchmarks for Rust aggregators
- Academic paper workspace and drafts (external to repo)

### Changed
- README rewritten to lead with Python/Flower usage
- `ByzantineAggregator` and `AggregationMethod` now derive `Serialize`/`Deserialize`

### Removed
- Unused QRES legacy dependencies: `blake3`, `curve25519-dalek`

## [0.2.0] - 2026-02-07

### Added
- Python bindings via PyO3
  - `ByzantineAggregator` class with NumPy support
  - Ergonomic string-based method selection ("trimmed_mean", "median", "fedavg")
  - Verified working: Byzantine clients successfully ignored in test
- PyPI trusted publishing
  - GitHub Actions workflow for automated releases
  - Maturin build system with abi3 compatibility (Python 3.8+)

### Changed
- Added `parse_method()` helper to convert string method names to enum
- Added `cdylib` crate type for Python extension module
- Added optional `pyo3` and `numpy` dependencies behind `python` feature

### Verified
- All 50 Rust tests pass
- Both Rust examples run successfully
- Python import and aggregation works locally
- Byzantine tolerance demonstrated in Python

## [0.1.0] - 2026-02-07

### Added
- Initial Rust release
- Byzantine-tolerant aggregation algorithms:
  - Trimmed mean (30% attack tolerance)
  - Median (coordinate-wise)
  - FedAvg (baseline)
- Quorum consensus implementation
- Comprehensive test suite
- Published to crates.io
