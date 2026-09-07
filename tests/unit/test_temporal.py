"""Phase 8 tests: temporal signals and the Temporal Risk Score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cloudsentinel.temporal import (
    TEMPORAL_COLUMNS,
    attach_temporal,
    behavioral_drift,
    cusum_change_points,
    ewma_deviation,
    rolling_zscore,
    seasonal_deviation,
)
from cloudsentinel.temporal.series import MIN_HISTORY, expanding_shifted_stats


def flat_series(n: int = 80, value: float = 10.0) -> pd.Series:
    return pd.Series([value] * n, dtype=float)


def test_ewma_stays_quiet_on_a_flat_series() -> None:
    assert ewma_deviation(flat_series()).abs().max() < 1e-6


def test_ewma_detects_a_slow_ramp_a_threshold_would_miss() -> None:
    """The exfiltration scenario ramps rather than spikes — this is the signal.

    Nothing in the ramp exceeds 2.6x the baseline level, so a static threshold
    set anywhere useful would stay silent throughout.
    """
    ramp = pd.Series([10.0] * 120 + list(np.linspace(11, 26, 40)))
    deviation = ewma_deviation(ramp)
    assert deviation.max() > 3.0
    assert deviation.iloc[:120].abs().max() < 1.0


def test_ewma_sensitivity_degrades_with_short_history() -> None:
    """A documented limit, not a surprise: the reference is a weekly window.

    With less than a week of history the slow mean is itself dragged by the
    ramp, and the same drift registers at roughly half the deviation. Identities
    younger than a week are therefore weakly covered by this signal — the
    novelty features carry them instead.
    """
    short = pd.Series([10.0] * 40 + list(np.linspace(11, 26, 40)))
    long = pd.Series([10.0] * 120 + list(np.linspace(11, 26, 40)))
    assert 2.0 < ewma_deviation(short).max() < ewma_deviation(long).max()


def test_rolling_zscore_flags_a_spike() -> None:
    series = pd.Series([10.0] * 60 + [400.0])
    scores = rolling_zscore(series)
    assert scores.iloc[-1] > 5.0
    assert scores.iloc[:60].abs().max() < 3.0


def test_expanding_stats_exclude_the_current_row() -> None:
    frame = pd.DataFrame({"g": ["a"] * 5, "x": [1.0, 1.0, 1.0, 1.0, 99.0]})
    mean, _ = expanding_shifted_stats(frame, "x", ["g"])
    assert np.isnan(mean.iloc[0])
    assert mean.iloc[4] == 1.0


def test_seasonal_deviation_uses_the_hour_of_week_profile() -> None:
    """Night activity is normal for a machine that always runs at night."""
    stamps = pd.date_range("2026-03-02", periods=24 * 21, freq="1h", tz="UTC")
    values = np.where(stamps.hour == 3, 100.0, 5.0)
    frame = pd.DataFrame({"identity_id": "svc", "window_start": stamps, "event_count": values})

    deviation = seasonal_deviation(frame, "event_count")
    night = deviation[(frame["window_start"].dt.hour == 3) & (np.arange(len(frame)) > 24 * 7)]
    assert night.abs().max() < 1.0


def test_seasonal_deviation_flags_an_off_profile_hour() -> None:
    stamps = pd.date_range("2026-03-02", periods=24 * 21, freq="1h", tz="UTC")
    values = np.where(stamps.hour == 3, 100.0, 5.0).astype(float)
    values[-22] = 100.0  # a busy hour in a slot that is normally quiet
    frame = pd.DataFrame({"identity_id": "svc", "window_start": stamps, "event_count": values})

    deviation = seasonal_deviation(frame, "event_count")
    assert deviation.iloc[-22] > 3.0


def test_cusum_accumulates_evidence_of_a_level_shift() -> None:
    series = pd.Series([10.0] * 40 + [22.0] * 20)
    statistic = cusum_change_points(series)
    assert statistic.iloc[:40].max() < 3.0
    assert statistic.iloc[40:].max() > 5.0


def test_cusum_is_quiet_without_a_shift() -> None:
    rng = np.random.default_rng(0)
    assert cusum_change_points(pd.Series(rng.normal(10, 1, 200))).max() < 8.0


def test_cusum_never_looks_ahead() -> None:
    """Truncating the future must not change the past."""
    series = pd.Series([10.0] * 40 + [90.0] * 20)
    full = cusum_change_points(series)
    partial = cusum_change_points(series.iloc[:45])
    assert np.allclose(full.iloc[:45], partial)


def test_drift_ratio_tracks_sustained_growth() -> None:
    series = pd.Series(list(np.linspace(10, 10, 200)) + list(np.linspace(10, 60, 40)))
    assert behavioral_drift(series).iloc[-1] > 2.0


def test_attach_temporal_adds_bounded_scores() -> None:
    stamps = pd.date_range("2026-03-01", periods=200, freq="1h", tz="UTC")
    store = pd.DataFrame(
        {
            "identity_id": "user_1",
            "window_start": stamps,
            "event_count": np.r_[np.full(180, 10.0), np.full(20, 300.0)],
            "bytes_out": 1000.0,
            "unique_resources": 3.0,
            "sensitive_api_count": 0.0,
        }
    )
    out = attach_temporal(store)

    assert set(TEMPORAL_COLUMNS) <= set(out.columns)
    assert out["temporal_risk"].between(0, 1).all()
    assert out["temporal_risk"].iloc[-10:].mean() > out["temporal_risk"].iloc[:150].mean() + 0.2


def test_cold_start_windows_score_low_rather_than_alarming() -> None:
    stamps = pd.date_range("2026-03-01", periods=MIN_HISTORY - 1, freq="1h", tz="UTC")
    store = pd.DataFrame(
        {
            "identity_id": "new_user",
            "window_start": stamps,
            "event_count": 5.0,
            "bytes_out": 10.0,
            "unique_resources": 1.0,
            "sensitive_api_count": 0.0,
        }
    )
    assert attach_temporal(store)["temporal_risk"].max() == pytest.approx(0.0, abs=1e-6)


def test_seasonal_profile_stays_silent_until_it_has_samples() -> None:
    """A profile built from one prior observation is noise, not seasonality.

    On a 25-day dataset the median hour-of-week slot recurred 3 times and 68% of
    rows had fewer than 2 prior samples, so the component carried a quarter of
    the score on nothing.
    """
    import pandas as pd

    from cloudsentinel.temporal.series import (
        MIN_SEASONAL_SAMPLES,
        seasonal_deviation,
        seasonal_slot,
    )

    stamps = pd.date_range("2026-03-02", periods=24 * 21, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "identity_id": "id_a",
            "window_start": stamps,
            "event_count": 10.0,
        }
    )
    frame.loc[frame.index[-1], "event_count"] = 400.0

    deviation = seasonal_deviation(frame, "event_count")
    prior = (
        frame.assign(_slot=seasonal_slot(frame))
        .groupby(["identity_id", "_slot"], observed=True)
        .cumcount()
    )

    assert (deviation[prior < MIN_SEASONAL_SAMPLES] == 0.0).all()
    # Once the slot has history, a 40x jump has to register.
    assert deviation.iloc[-1] > 5.0


def test_weekday_slots_get_enough_history_to_be_usable() -> None:
    import pandas as pd

    from cloudsentinel.temporal.series import MIN_SEASONAL_SAMPLES, seasonal_slot

    stamps = pd.date_range("2026-03-02", periods=24 * 25, freq="1h", tz="UTC")
    frame = pd.DataFrame({"window_start": stamps})
    counts = frame.assign(_slot=seasonal_slot(frame)).groupby("_slot").size()
    assert counts.median() > MIN_SEASONAL_SAMPLES * 2


def test_activity_drops_do_not_score_like_spikes() -> None:
    """Absolute deviation made people going home look like an incident."""
    import pandas as pd

    from cloudsentinel.temporal import temporal_features

    stamps = pd.date_range("2026-03-02", periods=200, freq="1h", tz="UTC")
    base = pd.DataFrame(
        {
            "identity_id": "id_a",
            "window_start": stamps,
            "event_count": 50.0,
            "bytes_out": 1000.0,
            "unique_resources": 5.0,
            "sensitive_api_count": 0.0,
        }
    )

    quiet = base.copy()
    quiet.loc[quiet.index[-1], "event_count"] = 1.0
    loud = base.copy()
    loud.loc[loud.index[-1], "event_count"] = 900.0

    quiet_risk = temporal_features(quiet)["temporal_risk"].iloc[-1]
    loud_risk = temporal_features(loud)["temporal_risk"].iloc[-1]

    assert loud_risk > quiet_risk + 0.2
    assert quiet_risk < 0.2


def test_drift_component_actually_reaches_the_score() -> None:
    """The drift column was computed on every run and then never read."""
    import numpy as np
    import pandas as pd

    from cloudsentinel.temporal.risk import WEIGHTS, temporal_features

    assert "drift" in WEIGHTS

    stamps = pd.date_range("2026-03-02", periods=400, freq="1h", tz="UTC")
    ramp = np.concatenate([np.full(300, 20.0), np.linspace(20.0, 200.0, 100)])
    frame = pd.DataFrame(
        {
            "identity_id": "id_a",
            "window_start": stamps,
            "event_count": ramp,
            "bytes_out": 100.0,
            "unique_resources": 3.0,
            "sensitive_api_count": 0.0,
        }
    )
    out = temporal_features(frame)
    assert out["temporal_drift_ratio"].iloc[-1] > 1.5
    assert out["temporal_risk"].iloc[-1] > out["temporal_risk"].iloc[250]
