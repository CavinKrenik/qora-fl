"""
Scalability Benchmark for Qora-FL
==================================
Measures wall-clock time and peak memory for aggregation across varying
client counts and parameter dimensions.

Usage::

    pip install qora-fl
    python examples/scalability_bench.py
"""


import time
import tracemalloc
import argparse
import numpy as np
from qora import ByzantineAggregator

# Default configuration
CLIENT_COUNTS = [10, 50, 100, 200, 500, 1000]
PARAM_SIZES = [1_000, 100_000, 1_000_000]
METHODS = ["fedavg", "trimmed_mean", "median", "krum", "multi_krum"]
N_RUNS = 5


def generate_updates(n_clients, n_params, rng):
    """Generate random gradient updates as list of (1, n_params) arrays."""
    return [rng.standard_normal((1, n_params)).astype(np.float32) for _ in range(n_clients)]


def make_method_string(method, n_clients):
    """Build method string with appropriate Krum parameters."""
    if method == "krum":
        f_val = max(1, n_clients // 5)
        return f"krum:{f_val}"
    elif method == "multi_krum":
        f_val = max(1, n_clients // 5)
        m_val = max(1, min(3, n_clients - 2 * f_val - 2))
        return f"multi_krum:{f_val}:{m_val}"
    return method


def measure_single(method_str, updates):
    """Measure time and peak memory for one aggregation call."""
    agg = ByzantineAggregator(method_str, 0.2)

    tracemalloc.start()
    start = time.perf_counter()
    try:
        agg.aggregate(updates)
    except Exception as e:
        tracemalloc.stop()
        return None, None, str(e)
    elapsed = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return elapsed, peak_mem, None


def main():
    parser = argparse.ArgumentParser(description="Qora-FL Scalability Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run a reduced grid for CI/fast verification")
    args = parser.parse_args()

    # Quick mode configuration
    if args.quick:
        client_counts = [10, 50, 100]
        param_sizes = [1_000, 100_000]
        n_runs = 2
    else:
        client_counts = CLIENT_COUNTS
        param_sizes = PARAM_SIZES
        n_runs = N_RUNS

    rng = np.random.default_rng(42)

    print("Qora-FL Scalability Benchmark")
    print("=" * 95)
    print(
        f"{'Method':<16} {'Clients':>8} {'Params':>10} "
        f"{'Time(ms)':>12} {'Memory(MB)':>12} {'Note':>20}"
    )
    print("-" * 95)

    for n_params in param_sizes:
        for n_clients in client_counts:
            # Skip prohibitively large combinations (>2 GB of data)
            total_bytes = 4 * n_clients * n_params
            if total_bytes > 2_000_000_000:
                continue

            updates = generate_updates(n_clients, n_params, rng)

            for method in METHODS:
                # Skip krum/multi_krum if n < 5
                if method in ("krum", "multi_krum") and n_clients < 5:
                    continue

                method_str = make_method_string(method, n_clients)

                times = []
                peak_mems = []
                error = None

                for _ in range(n_runs):
                    t, mem, err = measure_single(method_str, updates)
                    if err:
                        error = err
                        break
                    times.append(t)
                    peak_mems.append(mem)

                if error:
                    print(
                        f"{method:<16} {n_clients:>8} {n_params:>10} "
                        f"{'ERROR':>12} {'':>12} {error[:20]:>20}"
                    )
                else:
                    med_ms = np.median(times) * 1000
                    peak_mb = np.max(peak_mems) / (1024 * 1024)
                    note = ""
                    if med_ms > 1000:
                        note = "SLOW"
                    elif med_ms < 10:
                        note = "<10ms"
                    print(
                        f"{method:<16} {n_clients:>8} {n_params:>10} "
                        f"{med_ms:>12.2f} {peak_mb:>12.1f} {note:>20}"
                    )

        print()

    # Summary
    print("=" * 95)
    print("Notes:")
    print("  - Memory measured via tracemalloc (Python-side allocations only)")
    print("  - Rust-side memory is not tracked; estimate ~4 bytes * clients * params")
    print("  - Krum/Multi-Krum scale O(n^2*d); consider Trimmed Mean for n > 500")
    print("  - See docs/PERFORMANCE.md for detailed complexity analysis")


if __name__ == "__main__":
    main()
