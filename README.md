# Qora-FL

**Quorum-Oriented Robust Aggregation for Federated Learning**

[![Crates.io](https://img.shields.io/crates/v/qora-fl.svg)](https://crates.io/crates/qora-fl)
[![PyPI](https://img.shields.io/pypi/v/qora-fl.svg)](https://pypi.org/project/qora-fl/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18513738-blue)](https://doi.org/10.5281/zenodo.18513738)
[![License](https://img.shields.io/crates/l/qora-fl.svg)](LICENSE-MIT)

## Problem

Federated learning systems are fragile under adversarial or faulty clients.
Standard aggregation methods silently fail under model poisoning, gradient manipulation, or non-IID drift.

Qora-FL provides Rust-backed robust aggregation methods for federated learning.
Each method has explicit assumptions and tolerance conditions, documented below.
The included experiments evaluate selected attacks with up to 30% adversarial
clients.

> [!IMPORTANT]
> Qora-FL is an experimental robust-aggregation toolkit. Its algorithms are
> implemented and tested, but the project has not received an independent
> security review. Robustness depends on each method's stated assumptions,
> threat model, and correct configuration.

## Currently implemented

Behavior available through tested public paths:

- FedAvg baseline, with optional sample-count weighting
- Coordinate-wise median
- Coordinate-wise trimmed mean
- Krum and Multi-Krum, with their preconditions enforced
- Input validation: dimensions, non-finite rejection, client-ID alignment
- Krum quorum enforcement (`n >= 2f + 3`)
- Multi-Krum selection-bound enforcement (`1 <= m <= n - 2f - 2`)
- Reputation tracking and participation gating
- Fail-closed behavior when gating rejects every client
- Reputation numeric invariants (scores finite and within `[0, 1]`)
- Rust API and Python bindings
- Flower-compatible strategy adapter
- Rust CI across three operating systems; Python CI including the adapter

## Experimental or planned

Present in the codebase but **not** wired into the normal aggregation path, or
not yet implemented:

- Reputation-weighted robust aggregation (cubic influence weighting exists as a
  utility only)
- Optional norm-bound filtering during aggregation
- Broader audit records
- Cross-platform bit-identical end-to-end aggregation
- Independent security review
- Real-world deployment validation
- Additional attack and dataset evaluations

## Architecture

```
Clients produce model updates
              │
              ▼
┌────────────────────────────────┐
│   Qora Robust Aggregator       │
│                                │
│  ┌──────────────────────────┐  │
│  │ Trimmed Mean             │  │
│  │ Median                   │  │
│  │ Krum       (n ≥ 2f+3)    │  │
│  │ Multi-Krum (n ≥ 2f+3,    │  │
│  │        1 ≤ m ≤ n-2f-2)   │  │
│  │ FedAvg      (baseline)   │  │
│  └──────────────────────────┘  │
└─────────────┬──────────────────┘
              │
   Deviation signal (not trust)
              │
              ▼
     ┌─────────────────┐
     │    Reputation   │
     │     Manager     │
     │                 │
     │ Scores / Gates  │
     └────────┬────────┘
              │
              ▼
        Global Update
```

Gating runs *before* aggregation and reputation is updated *after* it. Scores
do not weight accepted updates -- see [Reputation](#reputation).

## Determinism

Qora-FL uses integer arithmetic for Krum distance and score calculations after
BFP-16 encoding (block floating-point, 16-bit mantissas, per-vector shared
exponent). Identical BFP-encoded inputs therefore produce deterministic Krum
scores and rankings.

**Full cross-platform bit-identical aggregation is not currently guaranteed.**
The pipeline is only partly integer:

| Stage | Arithmetic |
|---|---|
| BFP-16 encoding | floating-point (`log2`, scaling, rounding) |
| Krum distances and scores | integer (i32 shifts, i64 accumulation, saturating) |
| Krum result | the selected original `f32` update |
| Multi-Krum result | mean of selected original updates, in `f32` |
| Median, trimmed mean, FedAvg | floating-point throughout |

So the determinism claim covers the scoring stage, not the end-to-end result.
Selection *rankings* are reproducible; the returned values are `f32` and inherit
whatever the encoding and averaging stages do.

## Reputation

Reputation is **not trust**. It is a deviation-derived signal measuring how far
a client's update sits from the aggregate that round.

**What it does today:**

- **Tracks** scores across rounds, persisted via JSON serialization
- **Gates** participation: clients below the configured threshold are excluded
  before aggregation
- **Fails closed**: if every identified client is below the threshold, the round
  returns `AllUpdatesRejected` rather than reinstating them
- **Maintains numeric invariants**: scores are finite and within `[0, 1]`, and
  invalid mutation inputs are rejected rather than clamped

**What it does not do yet:** cubic influence weighting (`min(rep^3, 0.8)`) and
its cap are available as utilities, and a caller may use those values directly.
The primary aggregation path does not currently consume them, so the cap does
not provide protection for the default aggregation workflow.
Reputation-weighted robust aggregation is roadmap work.

Details that affect configuration:

- Unknown clients start at the default score of 0.5. A ban threshold above 0.5
  therefore rejects every first-round participant -- which now fails the round
  rather than passing it through ungated.
- Through the Flower adapter, reputation is computed from the complete
  flattened model, and each client receives at most one update per round.
  The adapter performs no gating of its own.

---

## Quick Start (Python + Flower)

```bash
pip install qora-fl[flower]
```

```python
import flwr as fl
from qora import QoraStrategy

strategy = QoraStrategy(
    aggregation_method="trimmed_mean",
    trim_fraction=0.2,
    min_fit_clients=5,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

`QoraStrategy` inherits from `FedAvg`, so standard Flower parameters
(`fraction_fit`, `min_fit_clients`, `initial_parameters`, `accept_failures`,
`fit_metrics_aggregation_fn`) are accepted. It is a **compatibility adapter**,
not a behavioral replacement for every Flower built-in strategy.

What the adapter does:

- Flower is optional, installed via `qora-fl[flower]`. Supported range
  `flwr>=1.5,<2`; CI tests 1.5.0 and 1.32.1.
- Validates each client's complete model structure -- layer count, per-layer
  shape, floating-point dtype, finiteness -- and rejects the round on a
  mismatch rather than dropping the client.
- Flattens each complete model and aggregates it in a single call, then
  restores the original layer shapes. Aggregation happens in `float32`;
  `float64` input is converted and its extra precision is not preserved.
  Integer parameter arrays are rejected.
- Weights by `num_examples` for FedAvg. Robust methods treat each accepted
  client update as one participant.
- Honors `accept_failures`: reported failures refuse the round when it is False.
- Returns `{}` for metrics unless a `fit_metrics_aggregation_fn` is supplied.
  Reputation is available through `QoraStrategy.get_reputation`.
- Performs no gating of its own; reputation gating and updates happen once, in
  the Rust aggregator.

## Python (Standalone)

```bash
pip install qora-fl
```

```python
import numpy as np
from qora import ByzantineAggregator

agg = ByzantineAggregator("trimmed_mean", 0.3)

# 7 honest clients + 3 Byzantine attackers
updates = [np.array([[1.0, 2.0]], dtype=np.float32) for _ in range(7)]
updates += [np.array([[100.0, 200.0]], dtype=np.float32) for _ in range(3)]

result = agg.aggregate(updates)
# Result is [1.0, 2.0] -- attackers rejected
```

### Reputation Tracking

```python
from qora import ReputationManager

rep = ReputationManager(ban_threshold=0.2)
rep.reward("hospital_A", 0.02)

# Penalize repeatedly -- simulates multiple rounds of bad behavior
for _ in range(5):
    rep.penalize("hospital_bad", 0.08)

print(rep.is_banned("hospital_bad"))  # True (score dropped below 0.2)
print(rep.active_clients())           # Only non-banned clients

# Persist between server restarts
json_state = rep.to_json()
rep2 = ReputationManager.from_json(json_state, ban_threshold=0.2)
```

## Rust

```bash
cargo add qora-fl
```

```rust
use qora_fl::{ByzantineAggregator, AggregationMethod};
use ndarray::array;

let mut agg = ByzantineAggregator::new(AggregationMethod::TrimmedMean, 0.2);

let updates = vec![
    array![[1.0, 2.0]],     // Honest
    array![[1.1, 2.1]],     // Honest
    array![[0.9, 1.9]],     // Honest
    array![[1.05, 2.05]],   // Honest
    array![[100.0, 200.0]], // Byzantine
];

let result = agg.aggregate(&updates, None).unwrap();
// [[1.05, 2.05]] -- the outlier is trimmed away
```

Note the client count: `trim_fraction = 0.2` removes `ceil(n * 0.2)` values from
each end, so at least `2 * ceil(n * 0.2) + 1` updates must be supplied or the
call returns `InsufficientQuorum`.

### Individual Functions

```rust
use qora_fl::{trimmed_mean, median, fedavg};
use ndarray::array;

let updates = vec![array![[1.0]], array![[2.0]], array![[3.0]]];

let tm = trimmed_mean(&updates, 0.2).unwrap();
let med = median(&updates).unwrap();
let avg = fedavg(&updates, None).unwrap();
```

## Aggregation methods and their assumptions

`n` is the number of accepted client updates; `f` is the configured Byzantine
bound; `m` is the number of Multi-Krum candidates selected.

| Method | Enforced precondition | Assumption for robustness |
|---|---|---|
| `TrimmedMean` | `0.0 <= trim_fraction <= 0.5`, and at least one value must survive trimming | Depends on trim fraction, adversarial proportion, attack model, and the distribution of honest updates |
| `Median` | none beyond shared input validation | Strictly fewer than 50% adversarial values **per coordinate** |
| `Krum` | `n >= 2f + 3` | Blanchard et al. (2017) |
| `MultiKrum` | `n >= 2f + 3` and `1 <= m <= n - 2f - 2` | Blanchard et al. (2017) |
| `FedAvg` | weights, when supplied, must be finite and non-negative with a positive total | **None.** Conventional sample-weighted baseline; provides no Byzantine robustness by itself |

**Trimmed mean** removes the configured fraction of the smallest and largest
values independently in each coordinate. A trim fraction is *not* an attacker
percentage: `trim_fraction = 0.2` does not establish tolerance of 30% malicious
clients, and no such universal guarantee is claimed here.

**Median** requires an honest majority per coordinate. Exactly 50% adversarial
is not strictly fewer than half, so "tolerates 50%" would be wrong.

**Krum** rejects configurations violating `n >= 2f + 3` with
`InsufficientQuorum`. There is no best-effort path below quorum.

**Multi-Krum** rejects an explicit `m` outside `1..=n - 2f - 2` with
`InvalidMultiKrumSelection`. When `m` is omitted, Qora-FL uses
`min(3, n - 2f - 2)`; explicit unsafe values are rejected rather than silently
reduced.

## Experiments

Qora-FL includes an attack-evaluation script for comparing aggregation methods
across selected attacks and adversarial-client fractions. **The repository does
not currently publish verified benchmark results** from a pinned dataset
artifact and repeated multi-seed evaluation. Run the included experiment to
evaluate the methods in your own environment.

The script is [`examples/attack_evaluation.py`](examples/attack_evaluation.py).
Its configuration:

| | |
|---|---|
| Dataset | MNIST via OpenML, falling back to scikit-learn `load_digits` if unavailable |
| Model | 2-layer MLP, hidden size 128 |
| Clients | 10 |
| Rounds | 15 |
| Local epochs | 1 |
| Learning rate | 0.1 |
| Adversarial fractions | 10%, 20%, 30% |
| Attacks | label flip, gradient scaling, sign flip, additive noise, ALIE |
| Methods | FedAvg, trimmed mean, median, Krum, Multi-Krum |
| Seed | 42 |
| Repetitions | 1 per configuration |

Two limitations to be aware of when interpreting a run:

- **The dataset can change silently.** Which of the two datasets is used depends
  on whether the OpenML fetch succeeds at run time, so two runs on different
  machines may not be comparing the same thing.
- **One run per configuration.** A single seed gives no variance estimate, so
  differences between methods cannot be separated from run-to-run noise.

Publishing results would require pinning the dataset (or recording its
checksum) so substitution cannot happen silently, running multiple seeds,
saving machine-readable results, generating tables and plots from those saved
results, and recording the environment and commit hash. That is separate work
from this documentation pass.

### Aggregation overhead

Criterion benchmarks in [`benches/aggregation.rs`](benches/aggregation.rs) cover
10/50/100 clients at 1K/100K/1M parameters. Run them with `cargo bench` to
produce numbers for your own hardware; timings are machine-specific and are not
reproduced here as project claims.

### Reproducing

```bash
# Aggregation overhead (Python)
python examples/benchmark_overhead.py

# MNIST poisoning: FedAvg vs Qora-FL under 30% attack
pip install matplotlib scikit-learn
python examples/mnist_poisoning_demo.py

# Criterion benchmarks (Rust)
cargo bench
```

```bash
# Rust examples
cargo run --example quickstart
cargo run --example compare_methods
```

## Validation status

Qora-FL is currently validated through:

- the Rust unit and integration test suite
- Python binding and Flower adapter tests
- cross-platform Rust CI (Linux, macOS, Windows) and Python CI
- Flower compatibility tests at both ends of the supported version range
- algorithm preconditions enforced in code and covered by tests
- the included examples and benchmarks, under their documented configurations

This is engineering validation. It is **not** an independent security review,
and it does not establish robustness outside each algorithm's stated
assumptions. See [SECURITY.md](SECURITY.md) for the boundaries.

## Background

The aggregation algorithms in Qora-FL (trimmed mean, coordinate-wise median,
Krum, Multi-Krum) are standard Byzantine-robust methods from the federated
learning literature [1][2].

## Project history

Qora-FL grew out of aggregation and distributed-trust experiments explored in
the earlier QRES project ([CavinKrenik/QRES_RaaS](https://github.com/CavinKrenik/QRES_RaaS),
historical work). Design choices here -- deviation-derived reputation, the
BFP-16 scoring path, adaptive trim fraction, and the influence cap -- were
informed by those earlier experiments and long-horizon simulations.

Qora-FL is a separate implementation with its own tests, experiments, and
assumptions. Results or guarantees from QRES do not automatically apply to
Qora-FL, and none of the evidence in this repository is inherited from it.

## Roadmap

- Reputation-weighted robust aggregation
- Norm-bound verification as an opt-in pre-aggregation filter
- TensorFlow Federated adapter
- Formal verification experiments for the integer scoring path

## Requirements

- **Rust:** edition 2021, stable toolchain
- **Python:** >= 3.8 (bindings built with PyO3 abi3)
- **NumPy:** >= 1.21.0
- **Flower** (optional): `>=1.5,<2` (`pip install qora-fl[flower]`). Tested in
  CI at 1.5.0 and 1.32.1. `import qora` works without it; only
  `qora.QoraStrategy` requires it.

### Building from Source

```bash
# Rust
cargo build
cargo test
cargo doc --open

# Python bindings
cd bindings/python
pip install maturin
maturin develop --features python
```

## References

1. Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. *NeurIPS*.
2. Yin, D., Chen, Y., Ramchandran, K., & Bartlett, P. (2018). Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. *ICML*.
3. Krenik, C. (2025). QRES: Resource-Aware Agentic Swarm for Distributed Learning. *Zenodo*. [DOI: 10.5281/zenodo.18474976](https://doi.org/10.5281/zenodo.18474976) -- earlier project, cited as lineage rather than as evidence for Qora-FL.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
