"""Phase 5 tests: feature store, leakage safety and behavioural distance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cloudsentinel.features import (
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    LABEL_COLUMNS,
    aggregate_windows,
    build_feature_store,
    feature_matrix,
    novelty_features,
)
from cloudsentinel.features.distance import (
    MIN_BASELINE_WINDOWS,
    behavioral_distance,
    expanding_baseline,
)
from cloudsentinel.pipeline import run_pipeline
from cloudsentinel.schema import events_to_dataframe
from cloudsentinel.simulator import CloudSimulator, SimulationConfig

CONFIG = SimulationConfig(
    n_identities=30, days=18, events_per_identity_per_day=18, attack_ratio=0.3, seed=17
)


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    frame, _ = run_pipeline(events_to_dataframe(CloudSimulator(CONFIG).run().events), strict=False)
    return frame


@pytest.fixture(scope="module")
def store(events: pd.DataFrame) -> pd.DataFrame:
    return build_feature_store(events)


def test_every_declared_feature_is_produced(store: pd.DataFrame) -> None:
    assert set(FEATURE_COLUMNS) <= set(store.columns)
    assert store[list(FEATURE_COLUMNS)].isna().sum().sum() == 0


def test_feature_matrix_excludes_keys_and_labels(store: pd.DataFrame) -> None:
    matrix = feature_matrix(store)
    assert list(matrix.columns) == list(FEATURE_COLUMNS)
    assert not set(matrix.columns) & set(LABEL_COLUMNS)
    assert not set(matrix.columns) & set(KEY_COLUMNS)


def test_windows_are_one_row_per_identity_and_window(store: pd.DataFrame) -> None:
    assert not store.duplicated(subset=["identity_id", "window_start"]).any()
    assert store["window_start"].dt.minute.eq(0).all()


def test_aggregations_match_the_raw_events(events: pd.DataFrame) -> None:
    rows = aggregate_windows(events)
    assert rows["event_count"].sum() == len(events)
    assert rows["bytes_out"].sum() == events["bytes_out"].sum()


def test_novelty_is_one_on_the_first_window(events: pd.DataFrame) -> None:
    novelty = novelty_features(events).sort_values("window_start")
    first = novelty.groupby("identity_id").head(1)
    assert (first["new_country_ratio"] == 1.0).all()
    assert (first["unusual_api_ratio"] == 1.0).all()
    assert (first["unusual_hour_score"] == 1.0).all()


def test_novelty_decays_as_history_accumulates(events: pd.DataFrame) -> None:
    novelty = novelty_features(events)
    early = novelty[novelty["history_events"] < 20]["unusual_api_ratio"].mean()
    late = novelty[novelty["history_events"] > 200]["unusual_api_ratio"].mean()
    assert late < early


def test_baseline_never_sees_the_current_window() -> None:
    """The core leakage guarantee: statistics for row t exclude row t."""
    rows = pd.DataFrame(
        {
            "identity_id": ["a"] * 5,
            "event_count": [1.0, 1.0, 1.0, 1.0, 100.0],
        }
    )
    center, _scale, history = expanding_baseline(rows, ("event_count",))

    assert np.isnan(center.iloc[0, 0])  # nothing precedes the first row
    assert center.iloc[4, 0] == 1.0  # the spike is not in its own baseline
    assert list(history) == [0, 1, 2, 3, 4]


def test_cold_start_yields_no_score_instead_of_a_confident_zero() -> None:
    rows = pd.DataFrame({"identity_id": ["a"] * 5, "event_count": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scores = behavioral_distance(rows, ("event_count",))
    assert scores["behavioral_distance"].isna().all()
    assert (scores["baseline_windows"] < MIN_BASELINE_WINDOWS).all()


def test_distance_rises_for_a_deviation_from_the_identitys_own_baseline() -> None:
    values = [10.0] * 40 + [900.0]
    rows = pd.DataFrame({"identity_id": ["a"] * 41, "event_count": values})
    scores = behavioral_distance(rows, ("event_count",))["behavioral_distance"]
    assert scores.iloc[-1] > scores.iloc[20] + 0.3


def test_distance_is_bounded(store: pd.DataFrame) -> None:
    warm = store["behavioral_distance"].dropna()
    assert warm.between(0.0, 1.0).all()


def test_attack_windows_are_farther_from_baseline_than_benign(store: pd.DataFrame) -> None:
    """End-to-end signal check — the features must carry something.

    This asserts a direction, not a metric. Real numbers come from the
    experiments, on real runs.
    """
    warm = store.dropna(subset=["behavioral_distance"])
    attack = warm[warm["is_attack"]]["behavioral_distance"]
    benign = warm[~warm["is_attack"]]["behavioral_distance"]

    assert len(attack) >= 5
    assert attack.mean() > benign.mean() + 0.1
    assert attack.median() > benign.quantile(0.75)


def test_labels_are_attached_only_to_campaign_windows(store: pd.DataFrame) -> None:
    labelled = store[store["is_attack"]]
    assert labelled["campaign_id"].notna().all()
    assert store.loc[~store["is_attack"], "scenario"].isna().all()
