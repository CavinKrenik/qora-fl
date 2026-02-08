"""
Aggregation Overhead Benchmark
==============================
Measures aggregation time for qora-fl methods on simulated gradients.
Target: <10ms for typical federated learning model sizes.

Usage::

    pip install qora-fl
    python examples/benchmark_overhead.py
"""

import time

import numpy as np

from qora import ByzantineAggregator

SIZES = [
    ("1K params", (1, 1_000)),
    ("100K params", (1, 100_000)),
    ("1M params", (1, 1_000_000)),
    ("11M params (ResNet-18)", (1, 11_000_000)),
]
NUM_CLIENTS = [10, 50, 100]
METHODS = ["fedavg", "trimmed_mean", "median", "krum"]
N_WARMUP = 3
N_RUNS = 20


def benchmark(method, n_clients, shape):
    agg = ByzantineAggregator(method, 0.2)
    rng = np.random.default_rng(42)
    updates = [rng.standard_normal(shape).astype(np.float32) for _ in range(n_clients)]

    # Warmup
    for _ in range(N_WARMUP):
        agg.aggregate(updates)

    # Timed runs
    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter_ns()
        agg.aggregate(updates)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        times.append(elapsed_ms)

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "p50_ms": np.percentile(times, 50),
        "p99_ms": np.percentile(times, 99),
    }


def main():
    print("Qora-FL Aggregation Overhead Benchmark")
    print("=" * 80)
    print(
        f"{'Method':<18} {'Clients':>8} {'Size':>22} "
        f"{'Mean(ms)':>10} {'P50(ms)':>10} {'P99(ms)':>10}"
    )
    print("-" * 80)

    for size_name, shape in SIZES:
        for n in NUM_CLIENTS:
            for method in METHODS:
                result = benchmark(method, n, shape)
                print(
                    f"{method:<18} {n:>8} {size_name:>22} "
                    f"{result['mean_ms']:>10.2f} "
                    f"{result['p50_ms']:>10.2f} "
                    f"{result['p99_ms']:>10.2f}"
                )
        print()

    # Verify <10ms claim for reasonable FL sizes
    print("=" * 80)
    print("Verification: <10ms for 10 clients x 100K params")
    result = benchmark("trimmed_mean", 10, (1, 100_000))
    if result["mean_ms"] < 10.0:
        print(f"  PASS: {result['mean_ms']:.2f}ms mean")
    else:
        print(f"  FAIL: {result['mean_ms']:.2f}ms mean (target: <10ms)")


if __name__ == "__main__":
    main()
