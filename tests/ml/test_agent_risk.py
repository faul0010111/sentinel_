"""The agent layer: cadence realism, the score, and its measured failure."""

from __future__ import annotations

import pandas as pd
import pytest

from cloudsentinel.features.agent_risk import AGENT_RISK_COLUMNS, agent_risk
from cloudsentinel.features.agents import AGENT_FEATURES
from cloudsentinel.pipeline import run_pipeline
from cloudsentinel.schema import events_to_dataframe
from cloudsentinel.simulator import CloudSimulator, SimulationConfig


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    config = SimulationConfig(
        n_identities=40,
        days=10,
        events_per_identity_per_day=20,
        attack_ratio=0.3,
        seed=17,
        campaign_start_range=(0.35, 0.95),
    )
    frame, _ = run_pipeline(events_to_dataframe(CloudSimulator(config).run().events), strict=False)
    return frame


def test_machines_work_in_sessions_and_humans_do_not(events: pd.DataFrame) -> None:
    """Without session structure, cadence features measure nothing at all.

    The first version of the simulator drew every timestamp uniformly across the
    day for every identity, which made an agent indistinguishable from an
    analyst on timing.
    """
    benign = events[~events["is_attack"]].sort_values(["identity_id", "timestamp"])
    gaps = benign.groupby("identity_id")["timestamp"].diff().dt.total_seconds()
    frame = benign.assign(gap=gaps).dropna(subset=["gap"])

    medians = frame.groupby("identity_type")["gap"].median()
    assert medians["human"] > 300
    for machine in ("ai_agent", "ci_cd"):
        if machine in medians:
            assert medians[machine] < 120


def test_agent_features_are_produced_and_bounded(events: pd.DataFrame) -> None:
    from cloudsentinel.features import build_feature_store

    store = build_feature_store(events, with_graph=False)
    assert set(AGENT_FEATURES) <= set(store.columns)
    assert store["cadence_regularity"].between(0.0, 1.0).all()
    assert store["tool_entropy"].ge(0.0).all()


def test_cadence_separates_machines_from_humans(events: pd.DataFrame) -> None:
    from cloudsentinel.features import build_feature_store

    store = build_feature_store(events, with_graph=False)
    benign = store[~store["is_attack"]]
    by_type = benign.groupby("identity_type")["cadence_regularity"].median()

    assert by_type["human"] < 0.3
    machines = [value for name, value in by_type.items() if name != "human"]
    assert min(machines) > by_type["human"]


def test_agent_risk_is_bounded_and_ordered(events: pd.DataFrame) -> None:
    from cloudsentinel.features import build_feature_store

    store = build_feature_store(events, with_graph=False)
    scores = agent_risk(store)

    assert list(scores.columns) == list(AGENT_RISK_COLUMNS)
    assert scores["agent_risk"].between(0.0, 1.0).all()
    assert len(scores) == len(store)
    assert scores.index.equals(store.index)


def test_agent_risk_requires_the_agent_features() -> None:
    with pytest.raises(KeyError, match="agent features"):
        agent_risk(
            pd.DataFrame({"identity_id": ["a"], "window_start": [pd.Timestamp("2026-01-01")]})
        )


def test_agent_risk_reacts_to_a_shift_in_working_shape() -> None:
    """The score does move when an identity's shape changes.

    Worth pinning even though the detector performs badly end to end: the two
    are different claims, and this one is what the score is built to do.
    """
    rows = 60
    frame = pd.DataFrame(
        {
            "identity_id": "agent_a",
            "window_start": pd.date_range("2026-03-01", periods=rows, freq="1h", tz="UTC"),
            "tool_call_ratio": 1.0,
            "tool_entropy": 0.8,
            "new_tool_ratio": 0.0,
            "cadence_regularity": 0.9,
            "read_to_act_ratio": 0.9,
            "burst_index": 0.8,
            "sensitive_api_count": 0.0,
        }
    )
    frame.loc[frame.index[-1], ["new_tool_ratio", "read_to_act_ratio", "sensitive_api_count"]] = [
        1.0,
        0.1,
        4.0,
    ]

    scores = agent_risk(frame)["agent_risk"]
    assert scores.iloc[-1] > scores.iloc[-10] + 0.3
