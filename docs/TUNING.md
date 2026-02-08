# Tuning Guide

Guidance for configuring Qora-FL's aggregation methods and reputation system.

## Method Selection

| Scenario | Recommended Method | Parameters |
|----------|-------------------|------------|
| No adversaries expected | FedAvg | -- |
| Up to 20% adversaries | Trimmed Mean | `trim_fraction=0.2` |
| Up to 30% adversaries | Trimmed Mean | `trim_fraction=0.3` |
| Up to 50% adversaries | Median | -- |
| Determinism required | Krum | `f` = expected Byzantine count |
| Smooth + deterministic | Multi-Krum | `f` = expected Byzantine count, `m=3` |
| Unknown attack rate | Trimmed Mean + adaptive | `adaptive_trim=True` |

## Reputation System

### Decay Rate

Controls how fast reputation scores return to the default (0.5) each round.
Call `decay_reputations(rate)` once per round.

| Rate | Behavior | Use Case |
|------|----------|----------|
| 0.0 | No decay | Permanent reputation (not recommended) |
| 0.01 | Very slow recovery | Persistent attackers, long-lived deployments |
| 0.03 | Moderate recovery | **Recommended default** |
| 0.05 | Faster recovery | Transient faults, mobile clients |
| 0.10 | Very fast recovery | Highly dynamic environments |

**Trade-off**: Higher decay allows formerly-banned clients to rejoin faster, but also forgives genuine attackers sooner.

### Ban Threshold

Clients with reputation below this value are excluded from aggregation.

| Threshold | Behavior |
|-----------|----------|
| 0.0 | No banning (reputation for observation only) |
| 0.1 | Very permissive (only excludes extreme attackers) |
| 0.2 | **Recommended** -- effective filtering, low false-positive risk |
| 0.3 | Aggressive (may exclude noisy honest clients) |
| 0.4 | Very aggressive (risk of false positives) |

### Trim Fraction

For Trimmed Mean: fraction to trim from each end per coordinate.

| Fraction | Byzantine Tolerance | Notes |
|----------|-------------------|-------|
| 0.1 | ~10% | Minimal trimming |
| 0.2 | ~20-30% | Good default |
| 0.3 | ~30% | Safe for moderate attack rates |
| 0.4 | ~40% | Maximum practical trimming |

### Adaptive Trim

When `adaptive_trim=True` with Trimmed Mean, the trim fraction is computed dynamically each round from the reputation distribution:

1. Count clients with reputation below 0.4 (suspicious threshold)
2. Compute `trim = suspicious_ratio + 0.05` (safety margin)
3. Clamp to `[min_trim, 0.49]` where `min_trim` = the constructor's `trim_fraction`

Use adaptive trim when:
- The attack fraction is unknown or changes over time
- Client IDs are provided (reputation tracking is active)
- Combined with `ban_threshold > 0` for defense-in-depth

## Reputation Internals

- **Default score**: 0.5 (neutral)
- **Reward**: +0.02 per round when update distance to aggregate < 1.0
- **Penalty**: -0.08 per round when update distance to aggregate > 10.0
- **Influence weight**: `min(rep^3, 0.8)` -- cubic scaling with cap prevents high-reputation clients from dominating
- **Fail-open**: If all clients would be banned, the ban filter is bypassed

## Running the Hyperparameter Study

```bash
python examples/reputation_study.py
```

This produces three tables:
1. Final accuracy across decay rate x ban threshold combinations
2. Rounds to ban all attackers for each configuration
3. Recovery accuracy after transient attacks at different decay rates
