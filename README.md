# Qora-FL

**Quorum-Oriented Robust Aggregation for Federated Learning**

[![Crates.io](https://img.shields.io/crates/v/qora-fl.svg)](https://crates.io/crates/qora-fl)
[![License](https://img.shields.io/crates/l/qora-fl.svg)](LICENSE-MIT)

Byzantine-tolerant aggregation for federated learning. Handles up to 30% malicious clients with minimal overhead.

## Features

- **Trimmed Mean** - Coordinate-wise outlier trimming (~30% Byzantine tolerance)
- **Median** - Coordinate-wise median (~50% Byzantine tolerance)
- **Krum** - Distance-based selection with fixed-point arithmetic (Blanchard et al., 2017)
- **FedAvg** - Standard baseline for comparison
- **Reputation tracking** - Automatically scores clients based on contribution quality
- **Deterministic consensus** - Optional I16F16 fixed-point math for bit-perfect reproducibility

## Quick Start

```bash
cargo add qora-fl
```

```rust
use qora_fl::{ByzantineAggregator, AggregationMethod};
use ndarray::array;

let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.3);

let updates = vec![
    array![[1.0, 2.0]],   // Honest
    array![[1.1, 2.1]],   // Honest
    array![[0.9, 1.9]],   // Honest
    array![[100.0, 200.0]], // Byzantine attacker
];

let result = agg.aggregate(&updates, None).unwrap();
// Result is close to [1.0, 2.0], attacker ignored
```

## Aggregation Methods

| Method | Byzantine Tolerance | Use Case |
|--------|-------------------|----------|
| `TrimmedMean` | ~30% of clients | Default choice for most FL deployments |
| `Median` | ~50% of clients | When stronger robustness is needed |
| `Krum` | n >= 2f+3 | Fixed-point deterministic consensus |
| `FedAvg` | None | Baseline comparison only |

## Using Individual Functions

```rust
use qora_fl::{trimmed_mean, median, fedavg};
use ndarray::array;

let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];

let tm = trimmed_mean(&updates, 0.2).unwrap();
let med = median(&updates).unwrap();
let avg = fedavg(&updates, None).unwrap();
```

## Examples

### Quickstart
```bash
cargo run --example quickstart
```

### Compare Methods
```bash
cargo run --example compare_methods
```

See [examples/](examples/) for more.

## Background

Core algorithms validated in [QRES](https://github.com/CavinKrenik/RaaS) (181-day autonomous IoT deployment with 30% Byzantine tolerance). Qora-FL adapts this proven consensus to standard federated learning workflows.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
