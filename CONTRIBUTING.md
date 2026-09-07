# Contributing

Thanks for your interest in CloudSentinel.

## Getting set up

```bash
make setup        # venv + editable install + pre-commit hooks
make test         # test suite
make lint         # ruff check + format check
```

## Ground rules

1. **No invented results.** Every metric in `reports/` must come from a real
   run of `make experiments`. Experiments on synthetic telemetry must carry the
   `SYNTHETIC DATA EXPERIMENT` label.
2. **No dead code.** Functions are implemented or not merged. Interfaces that
   wait for cloud infrastructure must ship with a working local implementation.
3. **Tests travel with the code.** New detectors need at least one test proving
   they flag their target attack scenario in the simulated dataset.
4. **Defensive scope only.** Contributions that add offensive capability
   (exploitation, credential theft, evasion tooling) will be rejected.

## Commit style

Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`,
`perf:`, `ci:`. Reference the build phase when relevant, e.g.
`feat(simulator): add lateral movement scenario (phase 3)`.

## Pull requests

- keep the CI green (lint, mypy, pytest, security workflow);
- describe the technical decision, not just the change;
- add an ADR under `docs/adr/` for architectural choices.
