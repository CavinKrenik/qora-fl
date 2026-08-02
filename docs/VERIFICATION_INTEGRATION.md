# Verification Integration — Design Note

Status: **implemented** on branch `feature/optional-norm-filtering`. Originally
written as research on `investigate/verification-integration` against `master`
at `411e357` (v0.3.1 + unreleased hardening work); the analysis below is
retained as the record of why the integration looks the way it does.

Two decisions changed when the design met the implementation. Both are recorded
inline where they were originally stated (§7) and summarized here:

1. **A norm rejection does not penalize reputation.** §7 originally said it
   should, once filtering was opt-in. That was reversed: a norm bound is a
   *participation filter*, not a reputation policy. A large norm is a policy
   violation, not evidence of malice, and folding it into the trust signal
   would mean an operator tightening a bound silently changes reputation
   dynamics — including banning clients for a limit they never agreed to. A
   rejected client is simply absent from the round, and absent from the
   distance-based scoring that follows. Making norm violations affect
   reputation remains available as a separately designed feature.
2. **`AllUpdatesRejected` carries only the submitted count.** It was introduced
   for reputation gating and held `{ total, rejected, threshold }`. Norm
   filtering feeds the same variant, and a round can mix the two policies, so no
   single threshold field can describe the outcome. Per-update reasons live in
   the audit record instead.

Everything else was implemented as ratified.

The [CHANGELOG](../CHANGELOG.md) deferred wiring `verification::norm_bound` and
`verification::audit` into the aggregation path on the grounds that it needs
policy decisions rather than code. This note answers the open questions and
states the policy, so that the wiring, when it happens, is executing a decision
instead of making one.

## Decisions (ratified 2026-07-29)

The findings below were reviewed and the following were adopted. Sections 1-7
are the supporting analysis; these four are the standing policy.

1. **The accumulated release is 0.4.0, not 0.3.2.** The Krum quorum change
   corrected a documented-contract violation, but downstream code can still
   observe `Some` -> `None`. For a pre-1.0 crate, `0.4.0` communicates that
   callers should review the upgrade rather than receiving it automatically
   under `^0.3`.
2. **`filter_by_norm_bound` is deprecated, not removed.** It stays in the public
   API behind a `#[deprecated]` attribute; removal is deferred to a later
   breaking release.
3. **`AuditLog` remains caller-owned and experimental.** It is not stabilized,
   not embedded in `ByzantineAggregator`, and not exposed to Python.
4. **Norm filtering is not integrated until its standalone behavior is
   corrected.** The f32 overflow in section 6 is a bug in shipped 0.3.1 code and
   is fixed on its own terms, independent of any integration.

No version metadata changes on this branch: `Cargo.toml` and `pyproject.toml`
stay at `0.3.1` with the work under `[Unreleased]` until the release is cut.

---

## 1. Current call sites

> Sections 1–7 describe the state **before** this integration. They are the
> supporting analysis, not a description of the code today.

**There are none.** `check_norm_bound`, `filter_by_norm_bound`, `AuditLog`, and
`AggregationAuditEntry` are referenced only by their own `#[cfg(test)]` modules
and by the `pub use` re-exports in [src/verification/mod.rs](../src/verification/mod.rs#L13-L15).

Verified across `src/`, `tests/`, `examples/`, `benches/`,
`bindings/python/qora/`, and `README.md`:

| Symbol | Non-test callers | Exposed to Python |
|---|---|---|
| `check_norm_bound` | none | no |
| `filter_by_norm_bound` | none | no |
| `AuditLog` / `AggregationAuditEntry` | none | no |
| `krum_condition_met` | [src/aggregators/krum.rs:12](../src/aggregators/krum.rs#L12) | indirectly |
| `krum_min_clients` | [src/aggregators/mod.rs:35](../src/aggregators/mod.rs#L35) | indirectly |
| `max_tolerable_f` | none (documented as a caller aid) | no |

So `verification` is currently two live functions (the Krum condition pair, wired
in Phase 6) and three dead ones. `src/lib.rs` exports `pub mod verification`
with no other integration.

## 2. Experimental utilities, or part of `ByzantineAggregator`?

**Today they are public experimental utilities, and the public surface is
accidental rather than designed.** Nothing in the crate documents them as
supported policy knobs; nothing consumes them; the Python bindings do not see
them. They were added in 0.3.1 as building blocks and never given an owner.

**Policy going forward, per symbol:**

- `check_norm_bound` — keep public. It is a pure predicate with an obvious
  contract and is genuinely useful to a caller implementing its own filtering.
  It must be fixed first (§6).
- `filter_by_norm_bound` — **do not wire it into aggregation; deprecate it, and
  do not remove it yet** (ratified). Its `is_ok()` filter is structurally
  incompatible with the Phase 6 policy that non-finite input is a hard error
  (§6). Silent index dropping is exactly the failure mode the hardening release
  removed elsewhere. It is public API, so removal waits for a later breaking
  release; `#[deprecated]` communicates the direction without breaking anyone.
- `AuditLog` — stays **caller-owned and experimental**, not a field of
  `ByzantineAggregator` (§4, §5), and not exposed to Python.

Norm bounding, if it ships, belongs on `ByzantineAggregator` as **opt-in**, in
the shape already used for ban gating: a constructor/setter (`Option<f32>`,
default `None`) plus a returned rejection report. It must not be mandatory —
the crate has no basis for choosing a bound (§3), and a wrong default silently
degrades every existing user's aggregation.

> **Shipped as** `with_norm_bound_filter(bound) -> Result<Self, QoraError>`,
> with `without_norm_bound_filter()` and a `norm_bound()` getter. The longer
> name says what it does: it *excludes* updates rather than merely verifying
> them. The rejection report is the audit record from
> `aggregate_with_audit`.

## 3. Who picks the bound

The caller, always. There is no defensible default: the correct L2 bound depends
on model scale, learning rate, local epoch count, and whether updates are deltas
or full weights — none of which the crate observes. A median-of-norms or
percentile-derived adaptive bound is a plausible future feature, but it is a
separate design (it is attacker-influenceable: enough colluding clients can
drag an adaptive bound upward), and it should not be the first version.

## 4. Audit entry contents

[`AggregationAuditEntry`](../src/verification/audit.rs#L10-L21) holds five
fields: `round`, `n_clients`, `n_excluded`, `method` (a `String`), and
`trim_fraction: Option<f32>`.

What it does **not** hold, and would need before it is useful for the stated
purpose ("reproducibility analysis and anomaly detection"):

- No timestamp.
- No client identities — neither the participants nor, more importantly, *which*
  clients were excluded. `n_excluded: 2` is not actionable.
- No rejection *reason*, so ban gating, norm rejection, and quorum failure would
  be indistinguishable once norm bounding exists.
- No record of the result (a hash or norm), so "reproducibility analysis" is not
  actually supported by the data captured.
- No `f`/`m` for Krum and Multi-Krum; `method` is a free-form `String`, so the
  parameters that determine the Byzantine guarantee are lost.
- No record of *failed* rounds. As written, `push` is only reachable on success,
  so the log would describe the rounds that went fine and omit the ones worth
  auditing.

Also note `round` is caller-supplied and never validated: a restored or
hand-written log can contain duplicate, out-of-order, or missing round numbers
(measured — see §5).

**Policy:** do not wire the current struct. Adding fields after it is in use is
a breaking change to any serialized log. Settle the schema first, and include at
minimum excluded client IDs with reasons, method parameters, and an outcome.

> **Amended 2026-07-30.** This section originally also called for a
> *timestamp*. That requirement is withdrawn: the library has no clock and no
> notion of a training round, so generating either would invent information and
> add a time dependency to the core crate. Timestamps, round numbers, and
> experiment identifiers are caller concerns -- a caller wraps the entry in
> their own record. The rest of the policy stands.
>
> Superseded by [`src/audit.rs`](../src/audit.rs), which implements the settled
> schema: `schema_version`, an effective-method descriptor, one
> `AggregationAuditDecision` per submitted update carrying its original index,
> optional client ID, and typed disposition, plus an `outcome` distinguishing
> `Aggregated` from `AllUpdatesRejected`.
>
> Entries record **the effective aggregation method parameters used for the
> attempt**, including the effective trim fraction and the resolved Multi-Krum
> selection count. Where runtime resolution occurs, the entry preserves both
> the requested configuration and the effective value. This is why the schema
> uses `AuditedAggregationMethod` rather than the configuration-only
> `AggregationMethod`: under `adaptive_trim` the fraction is recomputed each
> round and the configuration enum carries no fraction at all, and a bare
> `"multi_krum"` resolves `m` at aggregation time. A record naming only the
> configured method could not say what actually ran.
>
> The legacy `verification::audit` types are deprecated in favour of the new
> schema. They are not removed, and their serialized shape is **not**
> compatible with it.

## 5. Growth and persistence

**Unbounded.** `AuditLog::push` is a bare `Vec::push` with no cap, no ring
buffer, and no eviction; there is no `clear`, no `truncate`, and no
`drain`/`take`. Once a log is populated the only way to release the memory is to
drop the whole `AuditLog`. Nothing grows today only because nothing holds one.

This is the strongest argument against making `AuditLog` a field of
`ByzantineAggregator`: the aggregator is `Serialize`/`Deserialize` and is
persisted via `to_json`/`from_json` in the Python bindings, so an embedded log
would make the serialized aggregator state grow linearly and without bound in
the number of rounds — a long-running Flower server would eventually be unable
to checkpoint itself.

**Serialization, as measured:**

- Round-trips through `serde_json` correctly.
- `to_json` is gated behind `#[cfg(feature = "python")]`
  ([audit.rs:58](../src/verification/audit.rs#L58)) — an accident, not a design;
  a Rust-only user cannot serialize an audit log even though the type derives
  `Serialize`. There is no `from_json` counterpart at all.
- `entries` is private with no constructor from a `Vec`, so `Deserialize` is the
  only way to build a populated log externally — and it enforces nothing.
  Confirmed by probe: `{"entries":[{"round":9,...},{"round":2,...}]}` restores
  as a valid 2-entry log with rounds descending. **The "append-only" property
  holds only within a process; it does not survive a round trip.**
- Unknown fields are ignored (no `deny_unknown_fields`); missing fields are a
  hard error (`missing field 'n_excluded'`), and `{}` fails on `missing field
  'entries'`. So the format is forward-tolerant but not backward-tolerant — new
  fields will break restore of old logs unless they carry `#[serde(default)]`.

**Policy:** the log is caller-owned and caller-persisted. If it is ever attached
to the aggregator, it must be `#[serde(skip)]` and bounded (a `VecDeque` cap or
an explicit `drain` for callers that ship entries elsewhere). Restore must
either validate monotonic rounds or the doc comment must stop claiming
append-only. Un-gate `to_json` from the `python` feature and add `from_json`
regardless.

## 6. Does norm checking reject NaN independently?

**It rejects non-finite input, but by accident rather than by design, and the
mechanism is fragile.**

`l2_norm` ([src/math/norms.rs:7](../src/math/norms.rs#L7)) sums `x * x` and takes
`sqrt`. NaN propagates, `inf` propagates, and the comparison `norm <= max_norm`
is false for NaN — so the `else` branch fires. Measured:

| Input | Bound | Result |
|---|---|---|
| `[1.0, NaN]` | `10.0` | `Err("Update norm NaN exceeds bound 10.0000")` |
| `[1.0, inf]` | `10.0` | `Err("Update norm inf exceeds bound 10.0000")` |
| `[1e20, 1e20]` (true norm ≈ 1.41e20) | `1e30` | `Err("Update norm inf exceeds bound 1e30")` |
| `[1.0, 0.0]` | `NaN` | `Err("Update norm 1.0000 exceeds bound NaN")` |
| `[1.0, 0.0]` | `-1.0` | `Err(...)` |
| `0x0` array | `1.0` | `Ok(())` |

Four problems:

1. **The message is wrong.** "Update norm NaN exceeds bound 10.0" is not a true
   statement, and it misattributes a validation failure as a policy failure. A
   caller matching on `VerificationError` cannot tell the two apart — a real
   concern now that `QoraError::NonFiniteValue` exists as the correct variant.
2. **False rejection of legitimate finite input.** `l2_norm_sq` accumulates in
   `f32`, so any update whose squared norm exceeds `f32::MAX` (≈3.4e38) reports
   `inf` and is rejected regardless of the bound — the 1e20 row above has a true
   norm four orders of magnitude *below* its bound and is still refused. This is
   a live bug in `check_norm_bound` independent of any integration decision.
   Fix: accumulate in `f64`, or scale by the max absolute value before squaring.
3. **A non-finite `max_norm` rejects everything** with no distinct error.
4. **`filter_by_norm_bound` converts all of the above into a silent drop.** A
   NaN update is simply omitted from the returned indices (confirmed: indices
   `[0, 2]` for a 3-update batch whose middle entry is NaN), with no error and
   no record. That directly contradicts the Phase 6 policy.

**Policy:** norm checking must **assume finite input** and be documented as such.
`validate_updates` already runs first at every aggregation entry point and makes
non-finite input unreachable inside the aggregator, so the ordering is
`validate_updates` → norm bound → aggregate. `check_norm_bound` keeps a
defensive non-finite branch, but returns `QoraError::NonFiniteValue`, not
`VerificationError`, and the f32 overflow is fixed first.

## 7. What happens when every update fails verification?

Downstream behavior is already correct — `trimmed_mean`, `median`, and `fedavg`
all return `QoraError::EmptyUpdates` on an empty slice (measured) — so an
all-rejected round cannot produce a bogus aggregate. The question is what the
aggregator should do *before* reaching that point, and the crate currently
answers it inconsistently:

- Ban gating **fails open**: if every client is below `ban_threshold`, the
  filter is bypassed and all updates are used
  ([mod.rs:184-187](../src/aggregators/mod.rs#L184-L187)).
- Krum quorum failure **fails closed**: `InsufficientQuorum`, as of Phase 6.

**Policy: norm-bound rejection fails closed.** A round in which every update
exceeds the bound must return an error — a `QoraError` variant distinct from
`EmptyUpdates`, so the caller can distinguish "you sent nothing" from
"everything you sent was rejected." Rationale: unlike ban gating,
which reflects accumulated soft evidence and where failing open avoids a
reputation death-spiral stalling training, a universal norm violation means the
bound is misconfigured or the round is compromised. Silently proceeding in
either case is worse than stopping.

Two related sub-cases must also be decided at implementation time:

- **Partial rejection dropping below quorum.** If norm filtering leaves
  `n < 2f + 3` under Krum, the existing `InsufficientQuorum` error fires, but
  its `actual` will report the post-filter count with no indication that
  filtering caused it. The rejection report must make this legible.
- **Reputation interaction.** Does a norm rejection penalize?
  ~~**Yes, but only once norm bounding is opt-in and off by default**~~ —
  **reversed at implementation time: no.** See the note below.

  > **Amended 2026-07-30.** A norm bound is a participation filter, not a
  > reputation policy, and the two must not be conflated. The original argument
  > — that a rejected client "escapes scoring entirely", the same hole
  > non-finite input opened before Phase 6 — does not hold: non-finite input is
  > *malformed*, and refusing the whole round is the response, whereas an
  > over-bound update is well-formed and merely outside a limit the operator
  > chose. Penalizing it would mean tightening a bound silently changes trust
  > dynamics and can ban clients for a threshold they never agreed to, while
  > loosening it silently rehabilitates them.
  >
  > Implemented behavior: a rejected client is absent from the round and absent
  > from the distance-based scoring that follows. Its stored score does not
  > move, in either direction, however many rounds it is excluded for. Making
  > norm violations affect reputation stays available as a separately designed
  > feature.

---

## Summary of the policy

1. Norm bounding is **opt-in**, default off, `Option<f32>` on
   `ByzantineAggregator`; the caller picks the bound.
2. Order is `validate_updates` → norm bound → aggregate. Norm checking assumes
   finite input.
3. Fix `check_norm_bound` first: f64 accumulation, `NonFiniteValue` for
   non-finite input, explicit handling of a non-finite bound.
4. Do not integrate `filter_by_norm_bound`; deprecate it in place, removal
   deferred. Silent index dropping is not an acceptable failure mode in this
   crate.
5. All-rejected fails **closed** with a dedicated error variant. Rejected
   clients are excluded from distance-based scoring; they are **not** penalized
   for the rejection itself (amended — see §7).
6. `AuditLog` stays **caller-owned**; do not embed it in the serialized
   aggregator. Settle the entry schema (timestamp, excluded IDs + reasons,
   method parameters, outcome) before anything writes to it, and give it a bound
   and a drain.
7. Independent of the above: un-gate `AuditLog::to_json` from the `python`
   feature and add `from_json`.

## Prerequisites before any wiring

- [x] ~~Fix the `check_norm_bound` f32 overflow and error-variant issues.~~ —
      norms accumulate in `f64`; non-finite input returns `NonFiniteValue`, an
      unusable bound returns `InvalidNormBound`, and an over-bound update now
      returns the typed `NormBoundExceeded { norm, bound }` rather than a
      free-form `VerificationError` string.
- [x] ~~Decide the `AggregationAuditEntry` schema~~ -- settled and implemented
      in [`src/audit.rs`](../src/audit.rs) as a caller-owned, versioned record,
      and produced by `ByzantineAggregator::aggregate_with_audit`.
- [x] ~~Add the all-rejected `QoraError` variant.~~ —
      `AllUpdatesRejected { submitted }`, generic across both filtering
      policies.
- [x] ~~Decide whether `filter_by_norm_bound` is deprecated or removed~~ —
      deprecated in place, removal deferred to a later breaking release. The
      integration does not call it.

## What shipped

Implementation notes that the analysis above did not settle:

- **One engine.** `aggregate`, `aggregate_weighted`, `aggregate_with_audit`, and
  `aggregate_weighted_with_audit` all delegate to a single private `run`, so
  validation, filtering, effective-parameter resolution, and reputation updates
  each happen exactly once. The ordinary APIs discard the record; that is the
  only difference between them.
- **One aligned record per candidate.** Update, client ID, sample weight, and
  original index travel together in a `Candidate`, so filtering cannot leave
  them misaligned — the failure class already fixed once in the Flower adapter.
- **Effective parameters come from execution locals.** The trim fraction and the
  Multi-Krum selection count that the audit record reports are the same values
  passed to the aggregation call, not a second evaluation of the same formula.
- **Filtering order is part of the contract:** validate the whole batch →
  reputation gating → norm filtering on the survivors → fail closed → resolve
  effective parameters against the accepted cohort → enforce method
  preconditions against it → aggregate → update reputation for accepted clients.
  Malformed input is always an error rather than a filtering decision, and the
  first policy to reject an update owns its reason.
- **`check_norm_bound` and the filter share one comparison.** Both go through
  `evaluate_norm_bound`, which returns the measured norm on acceptance and a
  typed error on rejection, so the audit integration never parses a message and
  the two cannot disagree at the boundary (equality is acceptance).
- **The audit record's limits are documented rather than papered over.** A
  method-precondition failure after some candidates survived is neither
  `Aggregated` nor `AllUpdatesRejected`, so no entry is produced for it. Adding
  a failure outcome is a schema-version decision, not something to force into
  the existing two.
- **For an all-rejected round no method executed**, so the record carries **no
  effective parameters at all**: `effective_trim_fraction` and `effective_m` are
  `Option`, and both are `None`. Configuration is still reported in full — the
  configured trim fraction, the `adaptive` flag, `f`, and the `m` the caller
  requested are known regardless of what ran.

  An earlier draft reported the uncapped Multi-Krum default here, on the
  reasoning that the schema refuses a selection count below 1. That was wrong:
  it recorded a parameter that never executed, which is exactly what an audit
  record must not do. The `Option` is enforced in both directions —
  `Aggregated` with `None` and `AllUpdatesRejected` with `Some` are each
  rejected, on construction and on deserialization — so neither a missing
  measurement nor an invented one can enter a record.
- **Python receives configuration only.** `norm_bound` is a constructor
  argument on `ByzantineAggregator` and `QoraStrategy`; audit records are not
  exposed, so no binding design is frozen while the schema is experimental.

Implementation note for the deprecation: `#[deprecated]` on
`filter_by_norm_bound` will fire on the crate's own uses of it — the
`test_filter_by_norm_bound` unit test in
[src/verification/norm_bound.rs](../src/verification/norm_bound.rs#L67) and
possibly the `pub use` re-export in
[src/verification/mod.rs](../src/verification/mod.rs#L15). Since CI runs
`cargo clippy -- -D warnings`, those sites need `#[allow(deprecated)]` in the
same change or the build breaks. The test should be kept, not deleted: the
function still ships, so its behavior still needs pinning.
