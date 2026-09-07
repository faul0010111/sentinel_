"""Phase 1 tests: configuration loading, validation and severity mapping."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cloudsentinel import __version__
from cloudsentinel.settings import RiskSettings, RiskWeights, Settings, get_settings


def test_version_is_exposed() -> None:
    assert __version__.count(".") == 2


def test_default_config_file_loads() -> None:
    settings = get_settings()
    assert settings.project.name == "cloudsentinel"
    assert settings.simulator.n_identities > 0
    assert len(settings.simulator.scenarios) >= 10


def test_risk_weights_sum_to_one() -> None:
    settings = get_settings()
    assert sum(settings.risk.weights.as_dict().values()) == pytest.approx(1.0)


def test_risk_weights_reject_bad_sum() -> None:
    with pytest.raises(ValidationError):
        RiskWeights(behavioral=0.9, identity=0.9)


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0, "LOW"), (24, "LOW"), (25, "MEDIUM"), (49, "MEDIUM"), (74, "HIGH"), (91, "CRITICAL")],
)
def test_severity_bands(score: int, expected: str) -> None:
    assert RiskSettings().severity_for(score) == expected


def test_severity_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        RiskSettings().severity_for(101)


def test_severity_bands_must_be_contiguous() -> None:
    with pytest.raises(ValidationError):
        RiskSettings(severity_bands={"LOW": (0, 10), "HIGH": (50, 100)})


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CS_API__PORT", "9123")
    assert Settings().api.port == 9123


def test_business_hours_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(simulator={"seasonality": {"business_hours": (20, 8)}})
