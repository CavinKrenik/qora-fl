# Changelog

All notable changes to this project will be documented in this file.

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
