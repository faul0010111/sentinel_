"""The integrity harness, validated against pipelines broken on purpose.

Each test below reconstructs a defect that actually happened in this repository
and asserts the corresponding check catches it. A check that never fires is
decoration; these tests are what make the harness worth shipping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cloudsentinel.features import FEATURE_COLUMNS
from cloudsentinel.integrity import (
    Level,
    check_alert_budget,
    check_batch_independence,
    check_determinism,
    check_group_statistics,
    check_label_leakage,
    check_row_order_invariance,
    check_temporal_order,
    check_train_coverage,
)
from cloudsentinel.models.base import Split
from tests.ml.test_models import synthetic_store


def clean_store(rows: int = 600, seed: int = 2) -> pd.DataFrame:
    """A store with signal but no perfect separator.

    `synthetic_store` plants `new_country_ratio = 1.0` on exactly the attack
    rows — a flawless separator, which the leakage check correctly refuses. The
    harness found that in this project's own test fixture on its first run, so
    the fixture is what changes here, not the check.
    """
    rng = np.random.default_rng(seed)
    store = synthetic_store(rows=rows, seed=seed).sort_values("window_start").reset_index(drop=True)
    attacks = store["is_attack"].to_numpy()

    # Strong but overlapping: attacks are usually higher, benign rows sometimes are.
    store["new_country_ratio"] = np.clip(
        rng.normal(0.15, 0.12, rows) + attacks * rng.normal(0.5, 0.2, rows), 0.0, 1.0
    )
    store["behavioral_distance"] = np.clip(
        rng.normal(0.25, 0.1, rows) + attacks * rng.normal(0.45, 0.15, rows), 0.0, 1.0
    )
    # Same treatment for the other planted separators in the base fixture:
    # multiplying attack rows by 5000 leaves the two classes disjoint.
    store["bytes_out"] = np.exp(rng.normal(8.0, 1.2, rows) + attacks * rng.normal(2.0, 1.0, rows))
    store["privilege_change_count"] = rng.poisson(0.4 + attacks * 1.5, rows).astype(float)
    store["sensitive_api_count"] = rng.poisson(0.6 + attacks * 1.8, rows).astype(float)
    # A magnitude that grows over the window, so a prefix batch has a different
    # maximum than the whole — otherwise batch-relative scaling is invisible.
    store["ramp"] = np.linspace(1.0, 50.0, rows) * rng.uniform(0.9, 1.1, rows)
    return store


def ordered_split(rows: int = 600) -> Split:
    store = clean_store(rows)
    cut = int(rows * 0.6)
    return Split(train=store.iloc[:cut], test=store.iloc[cut:], strategy="temporal")


def honest_scores(split: Split) -> np.ndarray:
    """A well-behaved scorer: deterministic, per-row, order-independent."""
    return split.test["behavioral_distance"].to_numpy(dtype=float)


# ------------------------------------------------------- 1. temporal ordering
def test_temporal_order_passes_on_a_time_split() -> None:
    assert check_temporal_order(ordered_split()).level is Level.OK


def test_temporal_order_catches_a_random_split() -> None:
    store = clean_store(600)
    shuffled = store.sample(frac=1.0, random_state=0)
    split = Split(train=shuffled.iloc[:360], test=shuffled.iloc[360:], strategy="random")

    finding = check_temporal_order(split)
    assert finding.level is Level.FAIL
    assert finding.measurement is not None and finding.measurement > 0.2


# ---------------------------------------------------------- 2. label leakage
def test_label_leakage_is_quiet_on_ordinary_features() -> None:
    finding = check_label_leakage(ordered_split(), FEATURE_COLUMNS)
    assert finding.level is Level.OK


def test_label_leakage_catches_the_label_in_disguise() -> None:
    split = ordered_split()
    for side in (split.train, split.test):
        side["suspicious_score"] = side["is_attack"].astype(float) * 10.0

    finding = check_label_leakage(split, [*FEATURE_COLUMNS, "suspicious_score"])
    assert finding.level is Level.FAIL
    assert finding.detail["feature"] == "suspicious_score"


def test_label_leakage_catches_an_inverted_separator() -> None:
    """A perfectly inverted feature leaks exactly as hard as a direct one."""
    split = ordered_split()
    for side in (split.train, split.test):
        side["inverted"] = 1.0 - side["is_attack"].astype(float)

    assert check_label_leakage(split, ["inverted"]).level is Level.FAIL


# --------------------------------------------------------- 3. train coverage
def test_train_coverage_catches_a_feature_the_model_cannot_learn() -> None:
    """The defect that reversed Experiment 4: graph columns on 1 of 32 rows."""
    split = ordered_split()
    split.train["graph_risk"] = np.nan
    split.train.loc[split.train.index[:5], "graph_risk"] = 0.3
    split.test["graph_risk"] = 0.4

    finding = check_train_coverage(split, [*FEATURE_COLUMNS, "graph_risk"])
    assert finding.level is Level.FAIL
    assert finding.detail["feature"] == "graph_risk"
    assert finding.measurement is not None and finding.measurement < 0.5


def test_train_coverage_passes_when_both_sides_have_the_column() -> None:
    split = ordered_split()
    split.train["graph_risk"] = 0.3
    split.test["graph_risk"] = 0.4
    assert check_train_coverage(split, [*FEATURE_COLUMNS, "graph_risk"]).level is Level.OK


# ------------------------------------------------------- 4. group statistics
def test_group_statistics_catches_a_profile_built_from_nothing() -> None:
    """Hour-of-week slots on a short window: the phase-8 seasonality defect."""
    stamps = pd.date_range("2026-03-02", periods=24 * 12, freq="1h", tz="UTC")
    store = pd.DataFrame({"identity_id": "id_a", "window_start": stamps})
    store["hour_of_week"] = store["window_start"].dt.dayofweek * 24 + store["window_start"].dt.hour

    thin = check_group_statistics(store, ["identity_id", "hour_of_week"])
    assert thin.level is Level.FAIL
    assert thin.measurement is not None and thin.measurement > 0.5

    store["coarse_slot"] = (store["window_start"].dt.dayofweek >= 5).astype(int) * 24 + store[
        "window_start"
    ].dt.hour
    assert check_group_statistics(store, ["identity_id", "coarse_slot"]).level is Level.OK


# ----------------------------------------------------------- 5. alert budget
def test_alert_budget_catches_overspend_from_tied_scores() -> None:
    """The rule baseline spent 4.4% of a 1% budget this way."""
    labels = np.array([False] * 900 + [True] * 100)
    scores = np.concatenate([np.zeros(500), np.full(400, 0.5), np.full(100, 0.5)])

    finding = check_alert_budget(labels, scores, budget=0.01, threshold=0.5)
    assert finding.level is Level.FAIL
    assert finding.measurement is not None and finding.measurement > 0.4


def test_alert_budget_passes_when_the_threshold_is_chosen_properly() -> None:
    from cloudsentinel.detection import threshold_at_fpr

    labels = np.array([False] * 900 + [True] * 100)
    scores = np.concatenate([np.zeros(500), np.full(400, 0.5), np.full(100, 0.5)])
    threshold = threshold_at_fpr(labels, scores, 0.01)

    assert check_alert_budget(labels, scores, 0.01, threshold).level is Level.OK


# ------------------------------------------------------------ 6. determinism
def test_determinism_catches_an_unseeded_scorer() -> None:
    rng = np.random.default_rng()

    def unseeded(split: Split) -> np.ndarray:
        return rng.random(len(split.test))

    assert check_determinism(ordered_split(), unseeded).level is Level.FAIL
    assert check_determinism(ordered_split(), honest_scores).level is Level.OK


# --------------------------------------------------- 7. row order invariance
def test_row_order_invariance_catches_a_positional_scorer() -> None:
    """A scorer that returns results in its own order mispairs every label."""

    def positional(split: Split) -> np.ndarray:
        return np.linspace(0.0, 1.0, len(split.test))

    finding = check_row_order_invariance(ordered_split(), positional)
    assert finding.level is Level.FAIL
    assert check_row_order_invariance(ordered_split(), honest_scores).level is Level.OK


# --------------------------------------------------- 8. batch independence
def test_batch_independence_catches_batch_relative_scaling() -> None:
    """The phase-9 defect: centrality scaled by the maximum of the batch."""

    def batch_relative(split: Split) -> np.ndarray:
        values = split.test["ramp"].to_numpy(dtype=float)
        return values / max(values.max(), 1e-9)

    finding = check_batch_independence(ordered_split(), batch_relative)
    assert finding.level is Level.FAIL
    assert finding.measurement is not None and finding.measurement > 1e-9


def test_batch_independence_passes_for_a_per_row_scorer() -> None:
    assert check_batch_independence(ordered_split(), honest_scores).level is Level.OK


# ------------------------------------------------------------- full harness
def test_audit_reports_every_check_and_fails_loudly() -> None:
    from cloudsentinel.integrity import AuditInputs, audit

    split = ordered_split()
    scores = honest_scores(split)
    labels = split.test["is_attack"].to_numpy()

    clean = audit(
        AuditInputs(
            store=pd.concat([split.train, split.test]),
            split=split,
            score_fn=honest_scores,
            features=FEATURE_COLUMNS,
            labels=labels,
            scores=scores,
            threshold=float(np.quantile(scores[~labels], 0.999)),
            budget=0.01,
        )
    )
    assert len(clean.findings) == 8
    assert clean.passed, clean.summary()

    broken_split = ordered_split()
    broken_split.train["leak"] = broken_split.train["is_attack"].astype(float)
    broken_split.test["leak"] = broken_split.test["is_attack"].astype(float)
    broken = audit(
        AuditInputs(
            store=pd.concat([broken_split.train, broken_split.test]),
            split=broken_split,
            score_fn=honest_scores,
            features=[*FEATURE_COLUMNS, "leak"],
            labels=labels,
            scores=scores,
            threshold=0.0,
            budget=0.001,
        )
    )
    assert not broken.passed
    names = {finding.check for finding in broken.failures}
    assert {"label_leakage", "alert_budget"} <= names
