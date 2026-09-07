"""Phase 10 tests: risk components, fusion arithmetic and alert generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cloudsentinel.risk import COMPONENT_COLUMNS, RiskFusionEngine, build_components
from cloudsentinel.schema.enums import Severity
from cloudsentinel.settings import get_settings


def store(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "identity_id": [f"id_{index % 6}" for index in range(rows)],
            "identity_type": "human",
            "window_start": pd.date_range("2026-03-01", periods=rows, freq="1h", tz="UTC"),
            "behavioral_distance": np.linspace(0.0, 1.0, rows),
            "temporal_risk": 0.2,
            "graph_risk": 0.1,
            "new_country_ratio": 0.0,
            "new_ip_ratio": 0.0,
            "hosting_asn_ratio": 0.0,
            "failed_login_rate": 0.0,
            "unusual_hour_score": 0.0,
            "exfiltration_score": 0.1,
            "sensitive_objects_ratio": 0.0,
            "unusual_resource_ratio": 0.0,
            "download_velocity": 0.0,
            "privilege_level": 2.0,
            "privilege_change_count": 0.0,
            "sensitive_api_count": 0.0,
            "unique_resources": 3.0,
            "event_count": 20.0,
            "is_attack": False,
        }
    )


def test_components_are_bounded_and_complete() -> None:
    components = build_components(store())
    assert list(components.columns) == list(COMPONENT_COLUMNS)
    assert components.max().max() <= 1.0
    assert components.min().min() >= 0.0


def test_missing_layer_is_nan_not_zero() -> None:
    """A layer that did not run must not be scored as 'looked and found clean'."""
    frame = store().drop(columns=["graph_risk"])
    components = build_components(frame)
    assert components["graph"].isna().all()


def test_weight_is_redistributed_when_a_dimension_is_missing() -> None:
    full = store()
    without_graph = full.drop(columns=["graph_risk"])
    engine = RiskFusionEngine()

    scored_full = engine.score(full)
    scored_partial = engine.score(without_graph)

    assert (scored_full["risk_confidence"] == 1.0).all()
    # Confidence drops by exactly the missing dimension's weight.
    expected = 1.0 - get_settings().risk.weights.as_dict()["graph"]
    assert scored_partial["risk_confidence"].iloc[0] == pytest.approx(expected)
    assert scored_partial["risk_dimensions"].iloc[0] == len(COMPONENT_COLUMNS) - 1
    # Zero-filling would have dragged every score down; redistribution does not.
    assert scored_partial["risk_score"].iloc[-1] > scored_full["risk_score"].iloc[-1] * 0.9


def test_score_is_monotone_in_its_strongest_component() -> None:
    scored = RiskFusionEngine().score(store())
    assert scored["risk_score"].is_monotonic_increasing
    assert scored["risk_score"].between(0.0, 100.0).all()


def test_privilege_alone_cannot_produce_a_high_score() -> None:
    """Standing privilege must not make every admin window an alert."""
    frame = store(10)
    frame["behavioral_distance"] = 0.0
    frame["temporal_risk"] = 0.0
    frame["graph_risk"] = 0.0
    frame["exfiltration_score"] = 0.0
    frame["privilege_level"] = 5.0

    scored = RiskFusionEngine().score(frame)
    assert scored["risk_score"].max() < 25.0


def test_alert_explains_itself_and_matches_the_severity_band() -> None:
    frame = store()
    engine = RiskFusionEngine()
    scored = pd.concat([frame, engine.score(frame)], axis=1)

    alerts = engine.alerts(scored, threshold=float(scored["risk_score"].quantile(0.95)))
    assert alerts

    alert = alerts[0]
    assert alert.severity == Severity(get_settings().risk.severity_for(alert.risk_score))
    assert alert.risk_factors
    assert sum(factor.contribution for factor in alert.risk_factors) <= 1.0 + 1e-6
    assert alert.recommended_actions

    text = alert.explain()
    assert "WHY WAS THIS ALERT GENERATED?" in text
    assert "anomaly detected" not in text.lower()


def test_standing_components_are_labelled_in_the_explanation() -> None:
    frame = store(10)
    frame["privilege_change_count"] = 4.0
    engine = RiskFusionEngine()
    scored = pd.concat([frame, engine.score(frame)], axis=1)
    factors = engine.risk_factors(scored.iloc[-1])

    privilege = [factor for factor in factors if factor.name == "privilege"]
    assert privilege
    assert "standing exposure" in privilege[0].description


def test_calibration_moves_weight_toward_the_dimension_that_carries_signal() -> None:
    frame = store(200)
    frame["is_attack"] = frame["behavioral_distance"] > 0.9

    engine = RiskFusionEngine()
    before = dict(engine.weights)
    after = engine.calibrate(frame)

    assert sum(after.values()) == pytest.approx(1.0)
    assert after["behavioral"] > before["behavioral"]
    assert after["behavioral"] == max(after.values())


def test_calibration_without_positives_keeps_the_configured_weights() -> None:
    engine = RiskFusionEngine()
    original = dict(engine.weights)
    assert engine.calibrate(store()) == original
