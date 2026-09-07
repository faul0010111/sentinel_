"""Phase 2 tests: alerts are always explainable and internally consistent."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from cloudsentinel.schema import (
    Alert,
    AttackPathStep,
    IdentityType,
    RiskBreakdown,
    RiskCategory,
    RiskFactor,
    Severity,
)

WINDOW_END = datetime(2026, 3, 14, 10, 0, tzinfo=UTC)
WINDOW_START = WINDOW_END - timedelta(hours=1)

FACTORS = [
    RiskFactor(
        name="new_country_ratio",
        category=RiskCategory.BEHAVIORAL,
        contribution=0.4,
        observed=1.0,
        baseline=0.0,
        description="New country detected",
    ),
    RiskFactor(
        name="api_calls_per_minute",
        category=RiskCategory.TEMPORAL,
        contribution=0.3,
        observed=84.0,
        baseline=10.0,
        description="API activity increased",
    ),
]


def make_alert(**overrides: object) -> Alert:
    payload = {
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "identity": "user_001",
        "identity_type": IdentityType.HUMAN,
        "risk_score": 91.0,
        "confidence": 0.94,
        "risk_factors": FACTORS,
    }
    return Alert(**{**payload, **overrides})  # type: ignore[arg-type]


def test_severity_is_derived_from_score() -> None:
    assert make_alert().severity is Severity.CRITICAL
    assert make_alert(risk_score=30.0).severity is Severity.MEDIUM


def test_contradictory_severity_is_rejected() -> None:
    with pytest.raises(ValidationError, match="contradicts risk_score"):
        make_alert(severity=Severity.LOW)


def test_scored_alert_must_explain_itself() -> None:
    with pytest.raises(ValidationError, match="must list its risk factors"):
        make_alert(risk_factors=[])


def test_contributions_cannot_exceed_one() -> None:
    inflated = [factor.model_copy(update={"contribution": 0.8}) for factor in FACTORS]
    with pytest.raises(ValidationError, match="contributions exceed"):
        make_alert(risk_factors=inflated)


def test_inverted_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match="window_end"):
        make_alert(window_start=WINDOW_END, window_end=WINDOW_START)


def test_explain_is_human_readable() -> None:
    alert = make_alert(
        attack_path=[
            AttackPathStep(order=0, node_id="user_001", node_type="identity"),
            AttackPathStep(order=1, node_id="role_admin", node_type="role", edge="assumes"),
        ]
    )
    text = alert.explain()
    assert "WHY WAS THIS ALERT GENERATED?" in text
    assert "91/100" in text
    assert "New country detected" in text
    assert "user_001 -> role_admin" in text
    assert "8.4x baseline" in text


def test_risk_breakdown_fuses_with_configured_weights() -> None:
    breakdown = RiskBreakdown(
        behavioral=1.0, identity=1.0, graph=1.0, data_access=1.0, privilege=1.0, temporal=1.0
    )
    assert breakdown.fuse() == pytest.approx(100.0)

    partial = RiskBreakdown(
        behavioral=1.0, identity=0.0, graph=0.0, data_access=0.0, privilege=0.0, temporal=0.0
    )
    assert partial.fuse() == pytest.approx(25.0)


def test_detection_latency_is_measurable() -> None:
    alert = make_alert(created_at=WINDOW_END + timedelta(seconds=45))
    assert alert.detection_latency_seconds == pytest.approx(45.0)
