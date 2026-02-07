# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| < 0.3   | No        |

## Reporting a Vulnerability

Please report security vulnerabilities via [GitHub private security advisories](https://github.com/CavinKrenik/qora-fl/security/advisories/new).

Do not open public issues for security vulnerabilities.

## Scope

Qora-FL's security scope includes:

- Correctness of Byzantine-tolerant aggregation under documented threat models
- Integrity of the deterministic (Q16.16) execution path
- Soundness of the reputation gating mechanism

Qora-FL does **not** provide:

- Encrypted communication between FL clients and server
- Authentication of client identities
- Protection against Sybil attacks (multiple identities controlled by one adversary)
- Differential privacy guarantees on model updates

## Threat Model

Qora-FL assumes an honest-majority setting where up to 30% of participating clients may be Byzantine (arbitrary behavior). The aggregation algorithms are designed to produce correct results under this assumption. Exceeding the documented Byzantine tolerance bounds (e.g., >30% for trimmed mean, >50% for median) voids correctness guarantees.
