# Security Policy

## Scope and intent

CloudSentinel is a **defensive** security research project. It analyses cloud
telemetry to detect anomalous identity and workload behaviour. The attack
simulator produces **synthetic log records only** — it contains no exploit
code, no payloads and no capability to act against any real system.

## Supported versions

The `main` branch is the only supported version while the project is pre-1.0.

## Reporting a vulnerability

Please open a private security advisory on GitHub
(`Security` → `Report a vulnerability`) instead of a public issue.
Expect a first response within 7 days.

Include: affected component, reproduction steps, impact assessment and, when
possible, a suggested fix.

## Security controls in this repository

| Control | Tooling | Where |
|---|---|---|
| SAST | Bandit, CodeQL upload | `.github/workflows/security.yml` |
| Dependency audit | `pip-audit` | `security` workflow |
| Secret scanning | Gitleaks, `detect-secrets` (pre-commit) | workflow + hooks |
| Container scanning | Trivy | `security` workflow |
| IaC scanning | Trivy config (Terraform) | `security` workflow |
| SBOM | CycloneDX | `make sbom` / workflow artifact |
| Runtime hardening | non-root containers, pinned base images | `docker/` |

## Handling of cloud credentials

CloudSentinel never requires write permissions on a cloud account. Collectors
are designed for **read-only** roles (CloudTrail, IAM, VPC Flow Logs). No
credentials are committed to this repository; use `.env` (git-ignored) or your
cloud provider's identity federation.
