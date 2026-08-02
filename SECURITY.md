# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes       |
| 0.3.x   | No        |
| < 0.3   | No        |

0.3.x is superseded by 0.4.0, which fixes the input-validation and numeric
issues recorded in [docs/SECURITY_NOTES.md](docs/SECURITY_NOTES.md). Upgrading
is the fix; there are no backports.

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
- Fails closed when participation filtering rejects every submitted update,
  whether by reputation gating, by an optional norm bound, or by a mixture
- Enforces a caller-configured norm bound when one is set, excluding updates
  whose `f64` L2 norm exceeds it
- Records what each round decided about each submitted update, when the caller
  asks for it
- Provides implementations of trimmed mean, coordinate-wise median, Krum, and
  Multi-Krum

## Norm-bound filtering is a policy, not a detector

Optional norm filtering (0.4.0, disabled by default) enforces a **magnitude
policy the caller selected**. That is all it does.

It does **not**:

- Detect attacks. It has no model of adversarial behavior and makes no
  inference about intent.
- Establish that an accepted update is honest. Any update whose norm is under
  the bound passes, including a poisoned one crafted to stay under it.
- Establish that a rejected update is malicious. A large norm is a policy
  violation. Legitimate causes include a high learning rate, many local epochs,
  an unusually large local dataset, or a bound derived from a different model or
  training stage.
- Provide confidentiality, authentication, Sybil resistance, or secure
  transport. None of the items in the next section become available because a
  bound is configured.
- Change reputation. A norm rejection carries no reward, penalty, or ban.

Two operational risks it introduces:

- **Denial of service through misconfiguration.** A bound below honest update
  variation rejects honest clients. If it rejects all of them the round fails
  closed with `AllUpdatesRejected` and training stalls -- which is the intended
  failure direction, but it is still a stall. Derive the bound from observed
  honest norms before enforcing it; see [docs/TUNING.md](docs/TUNING.md).
- **Cohort reduction below quorum.** Method preconditions are checked against
  the accepted cohort, so filtering can push Krum or Multi-Krum below
  `n >= 2f + 3` and refuse a round that would otherwise have aggregated. An
  attacker who can predict the bound can deliberately trigger this by submitting
  over-bound updates.

## Audit records are observability, not evidence

`aggregate_with_audit` returns a record of what a round decided. It improves
observability and nothing more. Records are:

- **Not immutable.** A plain serializable struct the caller owns and can edit.
- **Not signed or hashed.** No integrity protection, no chaining, no ordering
  guarantee across rounds.
- **Not automatically persisted.** Qora-FL writes nothing to disk, keeps no log,
  and retains no entry inside the aggregator. Storage and retention are entirely
  the caller's.
- **Potentially privacy-relevant.** An entry carries whatever client identifiers
  the caller supplied and should be handled under the caller's own privacy
  policy. It carries no model data, no update values, and no hashes of either.

A record showing that a round rejected a client is a statement about what the
configured policy did, not an attestation that anything was verified.

## What Qora-FL does not provide

- Detection of every poisoning attack
- Detection of *any* attack by norm filtering, which enforces a magnitude policy
- Tamper-proof, signed, or automatically persisted audit logs
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

`n` is the **accepted** cohort -- what remains after reputation gating and norm
filtering, not what was submitted. Enabling either filter changes the `n` these
conditions are evaluated against.

Algorithmic robustness is not the same as security. These methods bound the
influence of deviating updates under stated statistical assumptions; they do
not authenticate participants, protect confidentiality, or resist an adversary
who controls enough of the cohort to satisfy none of the assumptions above.

`f` is a **configured** bound. If more than `f` clients are Byzantine, Krum's
guarantee does not hold regardless of what the code enforces.
