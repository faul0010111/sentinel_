"""Phase 6 tests: metrics, thresholds, latency and reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cloudsentinel.detection import (
    campaign_outcomes,
    evaluate_scores,
    run_comparison,
    threshold_at_fpr,
    write_reports,
)
from cloudsentinel.features import FEATURE_COLUMNS
from cloudsentinel.models import build_all, dataset_split
from tests.ml.test_models import synthetic_store


def test_threshold_respects_the_alert_budget() -> None:
    labels = np.array([False] * 990 + [True] * 10)
    scores = np.concatenate([np.linspace(0, 0.9, 990), np.full(10, 0.99)])
    threshold = threshold_at_fpr(labels, scores, 0.01)
    assert (scores[~labels] >= threshold).mean() <= 0.01


def test_budget_holds_when_scores_are_heavily_tied() -> None:
    """A coarse detector must not overspend the budget through ties."""
    labels = np.array([False] * 900 + [True] * 100)
    # 40% of benign windows share the same score as every attack window.
    scores = np.concatenate([np.zeros(540), np.full(360, 0.5), np.full(100, 0.5)])
    threshold = threshold_at_fpr(labels, scores, 0.01)
    assert (scores[~labels] >= threshold).mean() <= 0.01


def test_metrics_are_computed_from_the_budget_threshold() -> None:
    store = synthetic_store()
    scores = np.where(store["is_attack"], 0.95, 0.1)
    metrics = evaluate_scores("perfect", store, scores, fpr_budget=0.01)

    assert metrics.recall == 1.0
    assert metrics.false_negatives == 0
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.attack_windows == int(store["is_attack"].sum())


def test_a_useless_detector_scores_near_zero() -> None:
    store = synthetic_store()
    rng = np.random.default_rng(0)
    metrics = evaluate_scores("random", store, rng.random(len(store)))
    assert metrics.pr_auc is not None and metrics.pr_auc < 0.2
    assert metrics.recall < 0.5


def test_campaign_latency_is_floored_by_the_window() -> None:
    store = synthetic_store()
    scores = np.where(store["is_attack"], 0.99, 0.0)
    campaigns, detected, latency, per_scenario = campaign_outcomes(store, scores, 0.5, 60.0)

    assert campaigns == 1
    assert detected == 1
    assert latency is not None and latency >= 60.0
    assert per_scenario["data_exfiltration"] == 1.0


def test_undetected_campaign_reports_no_latency() -> None:
    store = synthetic_store()
    scores = np.zeros(len(store))
    campaigns, detected, latency, _ = campaign_outcomes(store, scores, 0.5, 60.0)
    assert campaigns == 1
    assert detected == 0
    assert latency is None


def test_comparison_runs_every_detector_on_the_same_split() -> None:
    split = dataset_split(synthetic_store(seed=1), synthetic_store(seed=2))
    table, metrics = run_comparison(split, build_all(), FEATURE_COLUMNS)

    assert len(table) == len(metrics) >= 5
    assert set(table.columns) >= {"detector", "precision", "recall", "pr_auc", "campaign_recall"}
    # The budget is a guarantee, ties included.
    assert table["false_positive_rate"].max() <= 0.01 + 1e-9


def test_detectors_needing_absent_columns_are_skipped_not_crashed() -> None:
    store = synthetic_store()
    split = dataset_split(
        store.drop(columns=["temporal_risk"]), store.drop(columns=["temporal_risk"])
    )
    table, _ = run_comparison(split, build_all(("temporal_risk", "rule_based")), FEATURE_COLUMNS)

    assert list(table["detector"]) == ["rule_based"]
    assert table.attrs["skipped"][0][0] == "temporal_risk"


def test_supervised_detectors_are_skipped_without_positive_labels() -> None:
    train = synthetic_store(seed=1)
    train["is_attack"] = False
    split = dataset_split(train, synthetic_store(seed=2))
    table, _ = run_comparison(split, build_all(("random_forest", "rule_based")), FEATURE_COLUMNS)
    assert list(table["detector"]) == ["rule_based"]


def test_reports_carry_the_synthetic_banner_and_provenance(tmp_path: Path) -> None:
    split = dataset_split(synthetic_store(seed=1), synthetic_store(seed=2))
    table, metrics = run_comparison(
        split, build_all(("rule_based", "isolation_forest")), FEATURE_COLUMNS
    )
    written = write_reports(table, metrics, tmp_path, provenance={"split": "independent-runs"})

    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    report = (tmp_path / "experiment_report.md").read_text()
    assert "SYNTHETIC DATA EXPERIMENT" in report
    assert "Accuracy is deliberately absent" in report
    assert pd.read_csv(tmp_path / "metrics.csv")["detector"].tolist()
    assert written["report"].exists()
