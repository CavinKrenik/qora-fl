# Changelog

All notable changes to this project will be documented in this file.

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
