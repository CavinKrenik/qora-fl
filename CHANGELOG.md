# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

Input validation hardening. Every item below is a correctness/security fix: in
each case the documented contract already existed and the implementation
contradicted it.

### Fixed

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
- `l2_norm` now uses an internal `f64` accumulator before returning `f32`,
  improving accuracy at extreme finite magnitudes.
- `l2_norm_sq` is documented as able to overflow to infinity or underflow to
  zero when the true squared norm falls outside `f32` range. This is a
  limitation of its return type rather than of its accumulation, so it cannot
  be removed without changing the signature; verification uses `l2_norm`
  instead.

### Added

- `src/validation.rs`: `validate_updates`, `validate_weights`,
  `validate_client_ids`, plus module documentation recording why non-finite
  input is rejected rather than sanitized. Crate-private, and located at the
  crate root rather than under `aggregators` because `verification` needs it
  too and `aggregators` already depends on `verification`.
- `verification::krum_min_clients(f)` -- minimum client count for a given `f`,
  with saturating arithmetic. Re-exported from `verification`.
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

- `verification::norm_bound` and `verification::audit` remain unwired from the
  aggregation path. Connecting them requires policy decisions (mandatory or
  opt-in norm checking; who selects the bound; whether rejection affects
  reputation; behavior when every update is rejected; whether the audit log is
  automatic, caller-owned, bounded, or persistent) and is out of scope here.

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
