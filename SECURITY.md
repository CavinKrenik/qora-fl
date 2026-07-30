# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| < 0.3   | No        |

## Reporting a Vulnerability

Please report security vulnerabilities via [GitHub private security advisories](https://github.com/CavinKrenik/qora-fl/security/advisories/new).

Do not open public issues for security vulnerabilities.

## Project status

Qora-FL is experimental and has **not received an independent security review**.
What follows describes implemented behavior and its limits, not an audited
guarantee.

## What Qora-FL does

Enforced in code and covered by the automated test suite:

- Rejects malformed input: dimension mismatches, non-finite values, and
  `client_ids` that do not align with the supplied updates
- Enforces Krum's `n >= 2f + 3` quorum condition, refusing configurations that
  violate it rather than proceeding best-effort
- Enforces Multi-Krum's selection bound `1 <= m <= n - 2f - 2`
- Maintains reputation numeric invariants: stored scores are finite and within
  `[0, 1]`, and invalid mutations are rejected without changing state
- Fails closed when reputation gating rejects every identified client
- Provides implementations of trimmed mean, coordinate-wise median, Krum, and
  Multi-Krum

## What Qora-FL does not provide

- Detection of every poisoning attack
- Correctness when an algorithm's assumptions are violated
- Confidentiality of client updates
- Authentication of client identities
- Secure transport
- Differential privacy
- Sybil resistance
- Cryptographic verification of updates
- Protection against a malicious *server*
- Protection against operator misconfiguration
- Production-readiness or independent audit

## Threat model

Each aggregation method carries its own assumption; there is no single
project-wide adversarial percentage.

| Method | Assumption |
|---|---|
| Trimmed mean | Depends on the configured trim fraction, adversarial proportion, attack model, and honest-update distribution. A trim fraction is not a tolerated attacker percentage. |
| Coordinate-wise median | Strictly fewer than 50% adversarial values per coordinate |
| Krum | `n >= 2f + 3`, with `f` an honest upper bound on Byzantine clients |
| Multi-Krum | `n >= 2f + 3` and `1 <= m <= n - 2f - 2` |
| FedAvg | None. Baseline only; no Byzantine robustness. |

Algorithmic robustness is not the same as security. These methods bound the
influence of deviating updates under stated statistical assumptions; they do
not authenticate participants, protect confidentiality, or resist an adversary
who controls enough of the cohort to satisfy none of the assumptions above.

`f` is a **configured** bound. If more than `f` clients are Byzantine, Krum's
guarantee does not hold regardless of what the code enforces.
