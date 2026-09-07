"""Phase 11 tests: attribution, coverage and the analyst narrative."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cloudsentinel.explainability import (
    Explanation,
    describe_feature,
    explain_detector,
    explain_window,
    identity_baselines,
    narrative,
)
from cloudsentinel.explainability.narrative import _format_comparison
from cloudsentinel.features import FEATURE_COLUMNS
from cloudsentinel.models import build
from tests.ml.test_models import synthetic_store


@pytest.fixture(scope="module")
def fitted() -> tuple[object, pd.DataFrame]:
    store = synthetic_store(rows=500, seed=4).reset_index(drop=True)
    detector = build("random_forest")
    detector.fit(store[list(FEATURE_COLUMNS)], store["is_attack"])
    return detector, store


def test_tree_detector_uses_shap(fitted: tuple[object, pd.DataFrame]) -> None:
    detector, store = fitted
    explanation = explain_detector(detector, store[list(FEATURE_COLUMNS)], sample=None)

    assert explanation.method == "shap.TreeExplainer"
    assert explanation.values.shape == (len(store), len(FEATURE_COLUMNS))
    assert list(explanation.values.columns) == list(FEATURE_COLUMNS)


def test_global_importance_ranks_the_planted_signal(fitted: tuple[object, pd.DataFrame]) -> None:
    detector, store = fitted
    explanation = explain_detector(detector, store[list(FEATURE_COLUMNS)], sample=None)
    top = list(explanation.global_importance().head(6).index)

    # The fixture plants its attack signal in these columns.
    assert {"bytes_out", "new_country_ratio", "privilege_change_count"} & set(top)


def test_family_importance_sums_over_all_features(fitted: tuple[object, pd.DataFrame]) -> None:
    detector, store = fitted
    explanation = explain_detector(detector, store[list(FEATURE_COLUMNS)], sample=None)
    families = explanation.family_importance()

    assert families.sum() == pytest.approx(explanation.global_importance().sum())
    assert (families >= 0).all()


def test_must_include_guarantees_alerts_are_explainable(
    fitted: tuple[object, pd.DataFrame],
) -> None:
    """An alert outside the sample would have no explanation at all."""
    detector, store = fitted
    alerts = np.flatnonzero(store["is_attack"].to_numpy())
    explanation = explain_detector(
        detector, store[list(FEATURE_COLUMNS)], sample=50, must_include=alerts
    )

    assert all(explanation.covers(int(position)) for position in alerts)
    assert len(explanation.positions) <= 50


def test_uncovered_row_fails_with_guidance(fitted: tuple[object, pd.DataFrame]) -> None:
    detector, store = fitted
    explanation = explain_detector(detector, store[list(FEATURE_COLUMNS)], sample=20, seed=1)
    missing = next(position for position in range(len(store)) if not explanation.covers(position))
    with pytest.raises(KeyError, match="must_include"):
        explanation.top_factors(missing)


def test_non_tree_detector_falls_back_and_says_so() -> None:
    store = synthetic_store(rows=120, seed=5).reset_index(drop=True)
    detector = build("rule_based")
    explanation = explain_detector(detector, store[list(FEATURE_COLUMNS)], sample=40)

    assert explanation.method.startswith("permutation")
    assert (explanation.values.iloc[0] == explanation.values.iloc[1]).all()


def test_factors_carry_observation_and_baseline(fitted: tuple[object, pd.DataFrame]) -> None:
    detector, store = fitted
    alerts = np.flatnonzero(store["is_attack"].to_numpy())
    explanation = explain_detector(detector, store[list(FEATURE_COLUMNS)], sample=None)
    factors = explain_window(explanation, int(alerts[0]), store)

    assert factors
    assert all(factor.observed is not None for factor in factors)
    assert sum(factor.contribution for factor in factors) <= 1.0 + 1e-6
    assert all(factor.description for factor in factors)


def test_narrative_never_says_only_anomaly_detected(fitted: tuple[object, pd.DataFrame]) -> None:
    detector, store = fitted
    alerts = np.flatnonzero(store["is_attack"].to_numpy())
    explanation = explain_detector(detector, store[list(FEATURE_COLUMNS)], sample=None)
    text = narrative(explanation, int(alerts[0]), store, score=91.0)

    assert "WHY WAS THIS ALERT GENERATED?" in text
    assert "shap.TreeExplainer" in text
    assert text.lower().count("anomaly detected") == 0
    assert len(text.splitlines()) > 8


def test_zero_baseline_is_reported_as_first_time_not_infinity() -> None:
    assert "none previously" in _format_comparison(5.0, 0.0)
    assert "10.0x" in _format_comparison(10.0, 1.0)
    assert "x)" not in _format_comparison(1.1, 1.0)


def test_identity_baseline_uses_only_that_identity() -> None:
    store = synthetic_store(rows=200, seed=6)
    store.loc[store["identity_id"] == "id_0", "bytes_out"] = 1000.0
    baselines = identity_baselines(store, "id_0", ["bytes_out"])
    assert baselines["bytes_out"] == 1000.0


def test_feature_phrases_cover_the_declared_features() -> None:
    unnamed = [name for name in FEATURE_COLUMNS if describe_feature(name)[0] == name]
    assert not unnamed, f"features with no analyst-facing phrase: {unnamed}"


def test_explanation_is_empty_when_no_feature_pushes_up() -> None:
    explanation = Explanation(
        detector="dummy",
        method="test",
        values=pd.DataFrame({"event_count": [-0.5], "bytes_out": [-0.2]}),
        features=pd.DataFrame({"event_count": [1.0], "bytes_out": [2.0]}),
        positions=np.array([0]),
    )
    store = pd.DataFrame(
        {
            "identity_id": ["id_0"],
            "identity_type": ["human"],
            "window_start": ["2026-03-01"],
            "event_count": [1.0],
            "bytes_out": [2.0],
        }
    )
    assert explain_window(explanation, 0, store) == []
    assert "review the detector" in narrative(explanation, 0, store, score=10.0)
