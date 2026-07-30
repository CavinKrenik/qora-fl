# Tuning Guide

Guidance for configuring Qora-FL's aggregation methods and reputation system.

## Method selection

| Scenario | Method | Parameters | Condition enforced in code |
|---|---|---|---|
| No adversaries expected | FedAvg | optional per-client weights | Weights finite, non-negative, positive total |
| Coordinate-wise trimming | Trimmed Mean | `trim_fraction` | `0.0 <= trim_fraction <= 0.5`, and at least one value must survive |
| Honest majority per coordinate | Median | -- | none beyond shared input validation |
| Single-vector selection | Krum | `f` | `n >= 2f + 3` |
| Averaged selection | Multi-Krum | `f`, optional `m` | `n >= 2f + 3` and `1 <= m <= n - 2f - 2` |
| Unknown attack rate | Trimmed Mean + adaptive | `adaptive_trim=True` | as trimmed mean |

`n` is the number of accepted updates; `f` is the configured Byzantine bound;
`m` is the number of Multi-Krum candidates selected.

### Choosing `f`

`f` is an assumption you supply, not something Qora-FL measures. If more than
`f` clients are actually Byzantine, the guarantee does not hold no matter what
the code enforces.

Krum requires `n >= 2f + 3`, so `f` is bounded by your expected cohort size:

| `f` | Minimum accepted updates |
|---|---|
| 0 | 3 |
| 1 | 5 |
| 2 | 7 |
| 3 | 9 |

Rounds below the requirement return `InsufficientQuorum`. Note that `n` is the
count *after* reputation gating, so gating can push a round below quorum.

### Choosing `m` for Multi-Krum

The safe maximum is `n - 2f - 2`. Worked examples:

| `n` | `f` | Valid `m` |
|---|---|---|
| 5 | 1 | 1 |
| 6 | 1 | 1..2 |
| 7 | 1 | 1..3 |
| 7 | 2 | 1 |
| 12 | 2 | 1..6 |

Omitting `m` (the bare `"multi_krum"` method string) selects
`min(3, n - 2f - 2)`, so small cohorts degrade gracefully. An **explicit** `m`
outside the valid range is rejected with `InvalidMultiKrumSelection` rather
than silently reduced.

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

Thresholds must be finite and within `[0, 1]`; `with_ban_threshold` and the
Python constructors reject anything else.

**Unknown clients start at 0.5.** A threshold above 0.5 therefore rejects every
first-time participant. Because gating now fails closed, that does not quietly
run the round ungated -- it returns `AllUpdatesRejected`. If every identified
client falls below the threshold in a later round, the same error is returned
rather than reinstating them.

### Sample weighting

`num_examples` weighting applies to **FedAvg only**, through the Flower adapter
or `aggregate_weighted`. The robust methods treat each accepted client update
as one participant: reweighting a median by a client-claimed sample count would
let an attacker buy influence with a number it controls. Supplying weights to a
non-FedAvg method returns `WeightsNotSupported`.

### Norm-bound filtering

`verification::check_norm_bound` exists and is tested, but **no aggregation
path invokes it**. It is not a configurable filter today; integrating it is
roadmap work. `filter_by_norm_bound` is deprecated.

### Trim fraction

For Trimmed Mean: the fraction removed from *each* end, per coordinate.
`ceil(n * trim_fraction)` values are dropped from both the low and high end, so
`2 * ceil(n * trim_fraction)` values are discarded in total and at least one
must remain -- otherwise the call returns `InsufficientQuorum`.

| Fraction | Values dropped per end (n=10) | Remaining |
|---|---|---|
| 0.1 | 1 | 8 |
| 0.2 | 2 | 6 |
| 0.3 | 3 | 4 |
| 0.4 | 4 | 2 |

**A trim fraction is not an attacker percentage.** Trimming 20% from each end
does not establish tolerance of 20% or 30% adversarial clients. How much
adversarial influence survives depends on the adversarial proportion, the
attack model, and how the honest updates are distributed in each coordinate.
Choose the fraction from how much of each coordinate's tail you are willing to
discard, and validate against your own threat model.

Invalid example: `trim_fraction = 0.3` with 4 updates drops 2 from each end,
leaving nothing, and returns `InsufficientQuorum`.

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
