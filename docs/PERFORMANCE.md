# Performance & Scalability Guide

## Algorithmic Complexity

| Method | Time Complexity | Space Complexity | Notes |
|--------|----------------|------------------|-------|
| FedAvg | O(n * d) | O(n * d) | Simple element-wise average |
| Trimmed Mean | O(n * d * log n) | O(n * d) | Per-coordinate sort, parallelized via rayon |
| Median | O(n * d * log n) | O(n * d) | Per-coordinate sort, parallelized via rayon |
| Krum | O(n² * d) | O(n² + n * d) | Pairwise BFP-16 distance matrix |
| Multi-Krum | O(n² * d) | O(n² + n * d) | Same as Krum + top-m sort |

Where **n** = number of clients, **d** = number of model parameters.

## Memory Estimation

Peak memory during aggregation (Rust side):

```
Linear methods:   ~4 bytes × n × d
Krum/Multi-Krum:  ~4 bytes × n × d  +  ~8 bytes × n²
```

Python overhead adds approximately 2x due to NumPy array copies across the FFI boundary.

### Example Estimates

| Clients | Parameters | Linear Methods | Krum/Multi-Krum |
|---------|-----------|----------------|-----------------|
| 10 | 100K | ~4 MB | ~4 MB |
| 50 | 100K | ~20 MB | ~20 MB |
| 100 | 100K | ~40 MB | ~40 MB |
| 100 | 1M | ~400 MB | ~400 MB |
| 500 | 1M | ~2 GB | ~2 GB |
| 1,000 | 1M | ~4 GB | ~4 GB |

## Recommended Client Limits

| Method | Practical Limit | Bottleneck |
|--------|----------------|------------|
| FedAvg | 10,000+ | Memory only |
| Trimmed Mean | 5,000+ | Per-coordinate sort |
| Median | 5,000+ | Per-coordinate sort |
| Krum | ~500 | O(n²) pairwise distances |
| Multi-Krum | ~500 | O(n²) pairwise distances |

For Krum/Multi-Krum beyond 500 clients, aggregation latency grows quadratically and may exceed acceptable round times. Consider Trimmed Mean or Median for larger deployments.

## Latency Targets

Qora-FL targets **< 10 ms** aggregation overhead for typical federated learning configurations (≤ 100 clients, ≤ 100K parameters). This ensures aggregation is negligible compared to client training time.

## Running Benchmarks

Rust (Criterion):
```bash
cargo bench
```

Python (latency):
```bash
python examples/benchmark_overhead.py
```

Python (scalability):
```bash
python examples/scalability_bench.py
```
