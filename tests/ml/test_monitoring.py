"""Phases 14, 16 and 17: tracking, drift, experiments and figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cloudsentinel.detection.experiments import EXPERIMENTS, verdict
from cloudsentinel.features import FEATURE_COLUMNS
from cloudsentinel.monitoring import Tracker, figures, monitor
from tests.ml.test_models import synthetic_store


def test_tracking_never_breaks_a_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An experiment must complete whether or not anything is tracking."""
    tracker = Tracker(experiment="tests", tracking_uri=f"file://{tmp_path}")
    with tracker.run("unit"):
        tracker.log_params({"seed": 1})
        tracker.log_metrics({"pr_auc": 0.9, "missing": None})  # type: ignore[dict-item]

    disabled = Tracker(experiment="tests", tracking_uri=f"file://{tmp_path}")
    disabled.enabled = False
    with disabled.run("disabled"):
        disabled.log_params({"seed": 2})
        disabled.log_metrics({"pr_auc": 0.5})
        disabled.log_artifact(tmp_path / "absent.csv")


def test_drift_is_quiet_when_nothing_moved() -> None:
    store = synthetic_store(rows=400, seed=1)
    report = monitor(store, store, score_column="behavioral_distance", threshold=0.5)

    assert report.drifted_features == []
    assert report.score_psi is not None and report.score_psi < 0.1
    assert not report.should_retrain()


def test_drift_fires_when_the_inputs_move() -> None:
    reference = synthetic_store(rows=400, seed=1)
    current = reference.copy()
    for column in ("bytes_out", "event_count", "objects_accessed", "api_calls_per_minute"):
        current[column] = current[column] * 400

    report = monitor(reference, current)
    assert len(report.drifted_features) >= 3
    assert report.should_retrain()
    assert "drifted" in report.summary()


def test_alert_rate_change_triggers_retraining() -> None:
    reference = synthetic_store(rows=400, seed=1)
    current = reference.copy()
    current["behavioral_distance"] = 0.95  # everything now clears the threshold

    report = monitor(reference, current, score_column="behavioral_distance", threshold=0.5)
    ratio = report.alert_rate_ratio
    assert ratio is not None and ratio > 2.0
    assert report.should_retrain()


def test_every_experiment_names_a_hypothesis_and_two_detectors() -> None:
    assert len(EXPERIMENTS) == 7
    numbers = [experiment.number for experiment in EXPERIMENTS]
    assert numbers == sorted(numbers)
    for experiment in EXPERIMENTS:
        assert experiment.hypothesis.startswith("H")
        assert experiment.left != experiment.right
        assert experiment.question.endswith("?")


def test_verdict_vocabulary_is_three_valued() -> None:
    assert verdict({"effect_size": 2.5}) == "consistent"
    assert verdict({"effect_size": 0.3}) == "within noise"
    assert verdict({"effect_size": float("nan")}) == "not evaluable"
    assert verdict({}) == "not evaluable"


def test_figures_are_written(tmp_path: Path) -> None:
    store = synthetic_store(rows=300, seed=3)
    labels = store["is_attack"].to_numpy()
    scores = np.where(labels, 0.9, 0.2) + np.random.default_rng(0).normal(0, 0.05, len(store))
    store["detector_score"] = scores
    store["window_start"] = pd.to_datetime(store["window_start"])

    summary = pd.DataFrame(
        {"detector": ["a", "b"], "pr_auc_mean": [0.9, 0.4], "pr_auc_std": [0.02, 0.05]}
    )
    per_run = pd.DataFrame(
        {"detector": ["a", "b"], "mean_detection_latency_minutes": [60.0, 120.0]}
    )
    importance = pd.Series({column: float(index) for index, column in enumerate(FEATURE_COLUMNS)})

    produced = [
        figures.model_comparison(summary, tmp_path / "comparison.png"),
        figures.precision_recall({"a": (None, scores)}, labels, tmp_path / "pr.png"),
        figures.roc({"a": (None, scores)}, labels, tmp_path / "roc.png"),
        figures.confusion(labels, scores > 0.5, tmp_path / "cm.png", "test"),
        figures.feature_importance(importance.sort_values(ascending=False), tmp_path / "fi.png"),
        figures.anomaly_timeline(store, "detector_score", tmp_path / "timeline.png"),
        figures.risk_distribution(store, "detector_score", tmp_path / "dist.png"),
        figures.detection_latency(per_run, tmp_path / "latency.png"),
    ]
    for path in produced:
        assert path.exists()
        assert path.stat().st_size > 5_000
