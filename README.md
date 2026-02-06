# Qora-FL: Quorum-Oriented Robust Aggregation for Federated Learning

> **Secure federated learning through quorum consensus.**

Qora (pronounced "KOR-ah") provides production-ready Byzantine tolerance for federated learning.

## Quick Start

```bash
cargo add qora-fl
```

```rust
use qora_fl::ByzantineAggregator;

let aggregator = ByzantineAggregator::new("trimmed_mean", 0.2);
// ...
```

## Status

🚧 **Work in Progress** - Adapting from QRES (IoT consensus) to FL aggregation.

## License

MIT OR Apache-2.0
