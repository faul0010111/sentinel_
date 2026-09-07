# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — closing the last honesty gaps
- **`streaming` was a one-line docstring** promising Kafka producers and
  consumers, while `docker-compose.yml` ran a worker on
  `cloudsentinel stream consume` — a command that did not exist. The scoring
  loop is now implemented: a pluggable source, a window buffer that releases a
  window only once it has closed, and late events dropped and counted rather
  than silently reopening a scored window. `KafkaSource` is declared as an
  interface and documented as never having run against a broker
- **`ingestion` was also empty**; it now re-exports the readers that were living
  elsewhere, so the layer named in the architecture is the layer that exists
- `cloudsentinel stream consume`, 7 streaming tests (270 collected, 269 passing, 1 skipped without a Kafka client)
- `docs/provenance.md` and `MANIFEST.sha256`: a full account of the
  unattributed files, what the audit found, and a checksum for every verified
  file so later changes are detectable

### Fixed
- replaying a Parquet dataset rejected **every** event and reported success on
  zero: the writer flattens labels into columns the event schema forbids. The
  replay now inverts the writer properly

### Added — non-human and AI-agent identity baselines
- session-based cadence in the simulator: machine identities now work in bursts
  at a near-constant interval, humans stay irregular. Every identity previously
  had uniformly scattered timestamps, so cadence features would have measured
  nothing — the fix had to come before the features
- `agent_risk`: a standalone score over the agent feature set, so the layer can
  be judged on its own rather than only as an XGBoost ablation
- `AgentRiskDetector` in the registry; 6 tests (263 total)

### Measured
- cadence regularity separates the population: humans 0.000, ai_agent 0.634,
  workload 0.643, service_account 0.679, ci_cd 0.806 (median, benign windows)
- `xgboost_agent` beats plain XGBoost by +0.032 PR-AUC, effect 0.81, 3/3 runs —
  small, consistent, at the edge of what three seeds support
- **`agent_risk` fails**: 0.040 PR-AUC against 0.683 for behavioural distance,
  0/3 runs. The same failure mode as risk fusion in phase 10 — a fixed linear
  combination of features dilutes, a trained model using them conditionally does
  not. Second independent occurrence of the pattern, kept in the repository as a
  negative result rather than tuned until it looked better

### Added — evaluation integrity harness
- `cloudsentinel.integrity`: eight model-agnostic checks that audit the
  evaluation rather than the model — temporal ordering, label leakage, training
  coverage, group-statistic depth, alert-budget compliance, determinism,
  row-order invariance, batch independence
- each check reproduces a defect that occurred in this repository, and each
  returns a measurement rather than a boolean
- `cloudsentinel audit` CLI, and `docs/evaluation-integrity.md` recording the
  six defects with their measured effects
- 15 tests, every one validating a check against a pipeline broken on purpose:
  a check that never fires is decoration

### Fixed
- the harness failed on its first run against this project's own test fixture:
  `synthetic_store` planted three perfect separators (`new_country_ratio` at
  exactly 1.0 on attack rows, `bytes_out` multiplied by 5,000, and the privilege
  counters). Any model evaluated against that fixture would have looked
  excellent for the wrong reason. The fixture was rewritten with overlapping
  distributions; the check was not relaxed

### Added — Phases 14-17: MLOps, DevSecOps, experiments and reports
- `monitoring/tracking.py`: MLflow wrapper that degrades to a no-op — a research
  repo whose experiments only run with a tracking server up is one most readers
  cannot run. Logs seed, split fractions, alert budget and dataset shape, since
  a metric without those is not reproducible
- `monitoring/drift.py`: data drift, concept drift and alert-rate drift kept
  separate because they fail at different times and have different remedies;
  the retraining trigger is deliberately conservative
- `detection/experiments.py`: the seven experiments as executable definitions,
  each a **paired** per-seed comparison with a three-valued verdict —
  consistent, within noise, not evaluable
- `monitoring/figures.py`: eight report figures, every comparison carrying its
  spread; PR curves lead and ROC is marked as optimistic at this prevalence
- `cloudsentinel experiments` CLI; results in `reports/experiments.csv`
- 7 tests (241 total)

### Fixed — DevSecOps pass
- **the leakage guard was three module-level asserts**, which `python -O` strips.
  The project's central invariant — no label or key may be declared as a feature
  — would have been silently absent in an optimized container. It now raises
  explicitly, verified under `python -O`
- MLflow silently disabled itself on the first real run: recent versions refuse
  the filesystem backend unless `MLFLOW_ALLOW_FILE_STORE` is set. Set explicitly,
  with the reason, since the SQLAlchemy alternative is absent from mlflow-skinny
- the prevalence annotation overlapped the legend in the PR-curve figure
- bandit: 6 low-severity findings resolved (1 real, 3 false positives on strings
  named like credentials, annotated); 0 medium, 0 high

### Fixed
- the simulator could place a campaign **past the end of the simulated window**,
  and since the window ends at the previous midnight, in the actual future — the
  schema rejected it and the whole run failed. At 0.95 of a 6-day run the offset
  plus 24h of jitter lands 6.7 days out. It only surfaced when the suite ran
  late in the day, which is the worst kind of bug to leave in a seeded
  simulator; campaign starts are now clamped inside the window

### Changed
- H2 no longer holds. Under two independent simulation runs the sequence
  autoencoder beat the dense one (0.712 vs 0.609); under the corrected
  reference/train/test split it loses 0 runs out of 3. The design changed, the
  conclusion changed with it, and both numbers stay on the record

### Added — Phase 13: dashboard
- Next.js 14 (app router) + TypeScript + Tailwind, seven views: alert queue,
  timeline, identities, attack paths, why-this-alert, models, results
- typed API client with one error path: a 503 from the API becomes an on-screen
  instruction (`cloudsentinel features build`, restart the API) rather than a
  spinner that never resolves
- attack paths rendered with Cytoscape, shared nodes drawn once so a resource
  several identities converge on reads as a hub
- risk dimensions show `n/a` for unmeasured ones and state the difference:
  not scored is not the same as scored clean
- results page carries the five-seed study with standard-deviation error bars
  and the note that the spreads are the size of the gaps between detectors
- `npm run build`, `tsc --noEmit` and `next lint` all clean; pages verified
  serving over HTTP

### Fixed
- `next/font/google` downloads the family at build time, so `npm run build`
  failed on any machine without access to fonts.googleapis.com. The family is
  linked at runtime instead, with a full fallback stack, so the build never
  depends on an external host

### Added — Phase 12: FastAPI
- all eleven endpoints from the brief: `/health`, `/metrics`, `/events`,
  `/predict`, `/alerts`, `/alerts/{id}`, `/identities`,
  `/identities/{id}/risk`, `/attack-paths`, `/models`, `/explanations/{id}`
- `DetectionService`: fits and scores once at startup, then serves from memory —
  fitting per request would make latency depend on dataset size and let two
  requests disagree about the same window
- a missing feature store returns **503 with the reason on `/health`**, not a
  crash at import: an empty data directory is an operational state, not a bug
- `/metrics` in Prometheus text format, hand-rolled rather than adding a
  dependency for six counters
- `/attack-paths` omits alerts whose path could not be reconstructed instead of
  returning an empty path, so "no path" cannot be read as "path of length zero"
- 15 API tests, including one asserting every alert has a retrievable
  explanation and one covering the not-ready path (234 total)

### Fixed
- building an alert then assigning its score raised a validation error against
  the *old* severity: `Alert` re-validates the pair on every assignment. The
  decision score is now a constructor argument (`build_alert(..., score=)`)
- a failed service load left `scored` and `alerts` populated while reporting
  `ready: false`, inviting a caller to trust one of them; partial state is now
  cleared

### Added — Phase 11: explainable AI
- SHAP attribution via `TreeExplainer` for the tree detectors, with a
  permutation-importance fallback for everything else — the method name travels
  with the explanation so a global-only approximation is never read as a local
  attribution
- global feature importance and a roll-up by feature family
- analyst narrative: every factor states the observed value **and** the
  identity's own baseline, so a reader can verify or dispute the claim; a zero
  baseline is reported as "none previously" rather than as an infinite ratio
- `must_include=` on `explain_detector`: alerts are guaranteed to be inside the
  sampled explanation, since an alert outside it would have no explanation at all
- `cloudsentinel explain`: importance tables plus narrated alerts, written to
  `reports/feature_importance.csv` and `reports/example_explanations.md`
- 12 tests, including one asserting every declared feature has an analyst-facing
  phrase (219 total)

### Fixed
- `explain` narrated alerts in row order, which showed two 9/100 false positives
  instead of the incidents; alerts are now ranked by score
- requesting an unsampled row raised an opaque `IndexError`; it now names the
  fix (`must_include=`)

### Added — Phase 10: risk fusion engine
- six risk dimensions assembled from the earlier layers, with an explicit split
  between **anomaly** dimensions (behavioural, temporal, graph novelty) and
  **standing** ones (privilege, data sensitivity), so an admin doing admin work
  cannot alert on privilege alone
- weighted fusion with **redistribution, not imputation**: a window missing the
  graph layer is scored on the five dimensions that exist, with confidence
  reduced by exactly the missing weight
- confidence reported separately from score — 90/100 from six dimensions and
  90/100 from two are different claims
- alert generation from the phase-2 schema: ranked risk factors, affected
  resources, prioritized recommended actions, and `explain()` output that never
  says only "anomaly detected"
- `RiskFusionEngine.calibrate`: weights fitted to each dimension's measured
  signal on training labels, exposed as a separate **supervised** detector so it
  is never mixed in with the label-free ones
- 9 tests for the layer (207 total)

### Fixed
- rounding each risk factor's contribution to four decimals could push their sum
  a hair above 1.0, which the Alert schema rejects by design; the rounding error
  is now absorbed in the largest factor

### Fixed — the evaluation split invalidated the graph result
- **The augmented model could not learn from the columns it was being tested
  on.** With the graph reference at 50% of the window and the train cut at 60%,
  only 9.3% of training rows carried a graph score and **1 of 32 training attack
  windows** had one. The reported "graph and temporal features make XGBoost
  worse (0.916 vs 0.942)" was a statement about the split, not about graphs.
- `reference_aware_split`: a three-way temporal split (reference / train / test)
  that drops the graph reference period from training. Training attack windows
  with graph scores went from 1/32 to 32/32.
- With the split corrected the sign reverses — the augmented model now scores
  **above** plain XGBoost — but a five-seed paired comparison puts the
  difference at +0.014 PR-AUC with a 0.049 spread and 3 wins out of 5, which is
  noise. The corrected conclusion is that this experiment cannot tell whether
  the structural signals help; not that they do.

### Added — repeated-seed evaluation
- `detection/repeat.py`: runs the full pipeline once per seed and reports mean
  and spread per detector, plus **paired** per-seed differences between two
  detectors, which removes the run-to-run variation that dominates at ~60 attack
  windows per test set
- `reports/variance_summary.csv`, `variance_per_run.csv`, `paired_differences.csv`
- regression tests for the split defect and for graph coverage in training

### Fixed — audit of the temporal and graph layers
Five defects found by auditing phases 8-9 against the project's own rules. Each
was measured before being fixed, and each now has a regression test.

- **Graph windows were scored against a graph containing them.** The reference
  graph was built from the first 50% of the same events and then used to score
  *all* of them, so `graph_new_edge_ratio` was structurally 0.000 for the 27,600
  windows inside the reference period — including the 31 attack windows among
  them — against 0.404 for attack windows outside it. Those windows are now left
  unscored (`NaN` plus a `graph_scored` flag), never zero-filled: 0.0 is the
  claim "reached nothing new", which is not what was measured.
- **Graph scores depended on the batch.** Betweenness was scaled by the maximum
  of whatever rows were being scored; 0.6% of rows changed by up to 0.144
  between two batches, and the score could not run online at all. It is now
  scaled by the reference graph's own peak.
- **The seasonal profile was estimated from one or two samples.** With
  hour-of-week slots on a 25-day dataset the median slot recurred 3 times and
  68% of rows had fewer than 2 prior observations. Slots are now
  (weekday/weekend, hour), and a slot stays silent until it has 3 prior samples.
- **Activity drops scored like spikes.** The saturation used absolute deviation,
  so benign windows where activity fell scored 0.274 against a benign mean of
  0.209 — mostly people going home. The risk score now counts increases only;
  signed deviations remain in the diagnostic columns.
- **The drift component was computed and never used.** Four passes over the
  driver features produced `temporal_drift_ratio`, which correlated 0.063 with
  the score it was documented as feeding. It is now a weighted component.

### Added
- `campaign_start_range` in the simulator. Campaigns were always injected in the
  last third of the window, which forced the comparison onto two independent
  runs — and left the graph layer unusable, since a graph fitted on one
  simulation cannot score another's identities. Spreading campaign starts makes
  a within-dataset temporal split viable for every layer at once.
- `Detector.can_run` plus a `skipped` record on the comparison table: a detector
  absent from the results now says why instead of vanishing.
- regression tests for all five defects (196 tests total).

### Added — Phase 8: temporal intelligence
- EWMA drift against a weekly reference, shifted rolling z-scores, hour-of-week
  seasonal deviation and online CUSUM change-point detection
- behavioural drift ratio (recent level vs longer-term level)
- **Temporal Risk Score** over four driver features, combined by max rather than
  mean so a single moving feature is not diluted
- `temporal_risk` detector in the comparison

### Added — Phase 9: graph analytics
- identity graph (identity, role, resource, API, host) with weighted edges
- degree and sampled betweenness centrality, Louvain communities, resource reach
- exposure per identity: distance to critical assets, blast radius, lateral
  movement potential, ranked within the fleet
- shortest suspicious paths and observed-path reconstruction for alerts
- **Graph Risk Score** with weight calibration that drops components carrying no
  variance in the reference graph and redistributes their weight
- `graph_risk` and `xgboost_graph_temporal` detectors
- temporal and graph columns are now attached during `features build`, so a
  store is self-contained
- 24 new tests (190 total)

### Fixed
- EWMA drift compared against a 24h mean, which the drift itself dragged along;
  a 2.6x ramp registered at ~1.3 sigma. The reference is now the weekly window
- the graph was fitted on one simulation and applied to another, where none of
  the identities exist: every edge looked new and the score degenerated to a
  constant. Reference and scoring are now split by time within one environment
- exposure metrics were divided by an arbitrary constant and saturated at 1.0
  for every identity; they are ranked within the fleet instead
- `run_comparison` crashed when a store lacked the temporal or graph columns;
  those detectors are now skipped and reported

### Added — Phase 7: deep learning
- dense Autoencoder, Variational Autoencoder and a GRU-based sequence
  Autoencoder, implemented directly on JAX arrays with explicit forward passes,
  losses and an Adam training loop
- `build_sequences`: per-identity sliding windows that never cross identities
  and never include anything after the window being scored; short histories are
  left-padded rather than dropped
- three new detectors behind the existing `Detector` interface, registered
  automatically when JAX is available
- ADR-0002 recording the backend decision
- 11 new tests (165 total)

### Changed
- `deep` extra is now `jax[cpu]`; `torch` moved to an optional `deep-torch`
  extra. The Linux `torch` wheel on PyPI pulls ~4 GB of CUDA dependencies for a
  workload that never leaves the CPU, and the CPU-only wheels are not on PyPI
- mypy targets 3.12 (jax stubs use PEP 695 syntax); the package still supports
  3.11 and CI tests both

### Fixed
- the GRU stored its hidden size as an int inside the parameter tree, which
  made `jax.grad` reject the whole structure; the size is now read off the
  parameter shapes

### Added — Phase 6: baseline models and evaluation
- one `Detector` contract (fit + score in 0..1) for rules, unsupervised and
  supervised models, with rank normalization of raw scores
- detectors: rule-based baseline, behavioral-distance-as-detector, Isolation
  Forest, LOF, One-Class SVM, Random Forest, XGBoost; registry with
  `available()` / `build()`
- temporal split and independent-run split (the latter is the default for the
  comparison, since campaigns are injected late in each simulation)
- scaler fitted on train only; quantile transform instead of standard scaling
- evaluation: precision, recall, F1, PR-AUC, ROC-AUC, FPR, FNR, campaign recall,
  per-scenario recall and detection latency; accuracy deliberately not reported
- alert-budget thresholding instead of F1-tuned thresholds
- reports: `model_comparison.csv`, `metrics.csv`, `scenario_recall.csv` and
  `experiment_report.md`, all carrying provenance and the synthetic banner
- CLI: `models list`, `evaluate`, `train`
- 23 new tests (154 total)

### Fixed
- the alert budget could be overspent by detectors with heavily tied scores: a
  quantile threshold lands on the tied value and flags every window sharing it
  (the rule baseline used 4.4% of a 1% budget). The threshold is now the
  smallest observed score whose realised FPR fits the budget

### Added — Phase 4: data pipeline
- dataset-level quality gate: required columns, duplicate events, null rates,
  future timestamps, collection gaps, value ranges, PSI drift
- normalization (types, casing, ordering) and per-event enrichment (hour,
  weekend, business hour, corporate IP, hosting ASN, sensitive action, data read)
- `run_pipeline` with a fail-loud strict mode and a permissive mode
- date-partitioned data lake writer
- CLI: `pipeline run`, `validate`

### Added — Phase 5: feature engineering
- 33 features across the four declared families, catalogued in `features/spec.py`
  with a leakage guard asserting labels and keys never enter the matrix
- identity x window feature store (1h windows) with vectorized aggregation
- leakage-safe novelty features via a single chronological pass over prior state
- **Behavioral Distance Score**: robust z-score + percentile deviation +
  Mahalanobis (weight redistributed, not imputed, when history is short),
  expanding and shifted so window *t* never sees itself
- cold-start windows return NaN instead of a confident zero
- CLI: `features build`, `features describe`

### Fixed
- PSI reported "stable" for zero-inflated columns (`bytes_out` is ~84% zeros);
  a dominant value now gets its own bin and the remainder is quantile-binned
- Mahalanobis distance crashed on single-feature inputs (`np.cov` returns a
  scalar, not a matrix)

### Performance
- feature build went from 57s to 1.6s on a 20k-event dataset (35x) by
  vectorizing the window aggregation and replacing per-window pandas calls in
  the novelty pass with a single array pass; outputs are unchanged

### Added — Phase 3: cloud attack simulator
- shared action catalog (34 actions) used by both normal traffic and attacks, so
  no scenario is separable by vocabulary alone
- synthetic estate (`build_environment`): buckets, objects, databases, secrets,
  compute, functions, pipelines, keys, roles and admin roles with sensitivity levels
- per-identity behavioural profiles across 8 personas and all 5 identity types
- normal generator with daily rhythm (working hours + lunch dip, nightly batch
  for machines), weekly and month-end seasonality, benign failures, occasional
  legitimate travel and VPN usage
- ten labelled scenarios, each staged along a kill chain, with `campaign_id`
- `CloudSimulator` with per-identity streaming, deterministic seeding and a
  two-pass target selection that guarantees scenario coverage
- `Manifest` ground truth (campaign windows and counts) written alongside the data
- CLI: `simulate`, `scenarios`; `make simulate` now runs end to end
- 39 new tests (109 total)

### Fixed
- `events_to_dataframe` failed on datasets mixing sub-second and whole-second
  timestamps (`pd.to_datetime` now uses `format="ISO8601"`)

### Added — Phase 2: unified event schema
- `cloudsentinel.schema` package: `CloudEvent`, `EventLabels`, `EventBatch`,
  `Alert`, `RiskBreakdown`, `RiskFactor`, `AttackPathStep`, `RecommendedAction`
- controlled vocabularies (`EventType`, `IdentityType`, `PrivilegeLevel`,
  `Sensitivity`, `AttackScenario`, `AttackStage`, `RiskCategory`, `Severity`)
- strict validation: UTC-aware timestamps with skew guard, ISO-3166 country and
  ASN normalization, IP validation, success/error-code consistency, extra fields
  rejected into `raw`
- label quarantine (`LABEL_FIELDS` + `CloudEvent.feature_payload()`) so ground
  truth cannot leak into features
- alert self-consistency: severity derived from the configured bands, factor
  contributions bounded, non-zero scores must be explained
- JSONL/Parquet round-trip, dataframe conversion and a non-raising batch
  validation gate (`ValidationReport`)
- CLI: `schema export|example|validate`; JSON Schema files in `docs/schemas/`
- 39 new tests (70 total)

### Added — Phase 1: repository foundation
- src-layout Python package `cloudsentinel` with all 14 documented layers
- typed configuration (`config/default.yaml` + `CS_*` env overrides) with
  validation of risk weights and severity bands
- `cloudsentinel` CLI (`version`, `config show|validate|init-dirs`)
- structured logging bootstrap (structlog, console/JSON)
- Makefile, Docker Compose stack (api, worker, dashboard, postgres, redpanda,
  mlflow, prometheus, grafana), API/dashboard Dockerfiles
- CI (`ruff`, `mypy`, `pytest`) and security pipeline (bandit, pip-audit,
  gitleaks, trivy image + IaC, SBOM), pre-commit hooks
- governance docs: README, SECURITY, CONTRIBUTING, CODE_OF_CONDUCT, ADR-0001
