"""Phase 6 tests: detector contract, splits and the rule baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cloudsentinel.features import FEATURE_COLUMNS
from cloudsentinel.models import (
    RuleBasedDetector,
    available,
    build,
    build_all,
    dataset_split,
    normalize_scores,
    temporal_split,
)


def synthetic_store(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {column: rng.lognormal(1.0, 0.4, rows) for column in FEATURE_COLUMNS},
    )
    for column in (
        "new_country_ratio",
        "hosting_asn_ratio",
        "unusual_resource_ratio",
        "failed_login_rate",
        "sensitive_objects_ratio",
        "unusual_hour_score",
        "new_ip_ratio",
        "new_service_ratio",
        "unusual_api_ratio",
        "unusual_region_ratio",
        "exfiltration_score",
    ):
        frame[column] = rng.uniform(0, 0.1, rows)

    attacks = rng.choice(rows, size=20, replace=False)
    frame.loc[attacks, "bytes_out"] *= 5000
    frame.loc[attacks, "new_country_ratio"] = 1.0
    frame.loc[attacks, "privilege_change_count"] = 5
    frame.loc[attacks, "sensitive_api_count"] = 6

    frame["is_attack"] = False
    frame.loc[attacks, "is_attack"] = True
    frame["identity_id"] = [f"id_{index % 10}" for index in range(rows)]
    frame["window_start"] = pd.date_range("2026-03-01", periods=rows, freq="1h", tz="UTC")
    frame["campaign_id"] = np.where(frame["is_attack"], "camp_1", None)
    frame["scenario"] = np.where(frame["is_attack"], "data_exfiltration", None)
    frame["behavioral_distance"] = np.where(frame["is_attack"], 0.9, 0.2)
    frame["temporal_risk"] = np.where(frame["is_attack"], 0.8, 0.15)
    frame["graph_risk"] = np.where(frame["is_attack"], 0.7, 0.25)
    return frame


def test_registry_exposes_every_detector() -> None:
    names = available()
    assert {"rule_based", "isolation_forest", "random_forest", "behavioral_distance"} <= set(names)
    assert all(build(name).name == name for name in names)


def test_unknown_detector_fails_clearly() -> None:
    with pytest.raises(KeyError, match="unknown detector"):
        build("magic_detector")


@pytest.mark.parametrize(
    "name", ["rule_based", "isolation_forest", "one_class_svm", "random_forest"]
)
def test_detectors_return_bounded_scores(name: str) -> None:
    store = synthetic_store()
    detector = build(name)
    labels = store["is_attack"] if detector.requires_labels else None
    detector.fit(store[list(FEATURE_COLUMNS)], labels)
    scores = detector.score(store[list(FEATURE_COLUMNS)])

    assert len(scores) == len(store)
    assert np.all((scores >= 0) & (scores <= 1))


def test_rule_baseline_fires_on_its_own_conditions() -> None:
    store = synthetic_store()
    scores = RuleBasedDetector().score(store[list(FEATURE_COLUMNS)])
    assert (
        scores[store["is_attack"].to_numpy()].mean() > scores[~store["is_attack"].to_numpy()].mean()
    )


def test_rules_do_not_learn() -> None:
    """A static baseline must produce identical scores before and after fit."""
    store = synthetic_store()
    detector = RuleBasedDetector()
    before = detector.score(store[list(FEATURE_COLUMNS)])
    detector.fit(store[list(FEATURE_COLUMNS)])
    assert np.array_equal(before, detector.score(store[list(FEATURE_COLUMNS)]))


def test_supervised_detector_refuses_to_fit_without_labels() -> None:
    store = synthetic_store()
    with pytest.raises(ValueError, match="requires labels"):
        build("random_forest").fit(store[list(FEATURE_COLUMNS)], None)


def test_temporal_split_never_overlaps_in_time() -> None:
    store = synthetic_store()
    split = temporal_split(store, 0.6)
    assert split.train["window_start"].max() < split.test["window_start"].min()
    assert len(split.train) + len(split.test) == len(store)


def test_dataset_split_keeps_runs_separate() -> None:
    split = dataset_split(synthetic_store(seed=1), synthetic_store(seed=2))
    assert split.strategy == "independent-runs"
    assert len(split.train) == len(split.test)


def test_rank_normalization_is_robust_to_one_extreme_value() -> None:
    scores = normalize_scores(np.array([1.0, 2.0, 3.0, 1e9]))
    assert scores.min() == 0.0
    assert scores.max() == 1.0
    assert scores[1] > 0.3  # min-max scaling would collapse this to ~0


def test_constant_scores_normalize_to_zero() -> None:
    assert np.all(normalize_scores(np.full(10, 0.5)) == 0.0)


def test_build_all_returns_every_registered_detector() -> None:
    assert len(build_all()) == len(available())
    assert [d.name for d in build_all(("rule_based",))] == ["rule_based"]


def test_reference_aware_split_keeps_three_periods_in_order() -> None:
    from cloudsentinel.models import reference_aware_split

    store = synthetic_store(rows=400)
    split = reference_aware_split(store, 0.3, 0.65)

    assert split.train["window_start"].max() < split.test["window_start"].min()
    # The reference period is excluded from training, not merged into it.
    assert len(split.train) + len(split.test) < len(store)
    reference_end = store["window_start"].quantile(0.3)
    assert split.train["window_start"].min() > reference_end


def test_augmented_detector_gets_graph_features_it_can_learn_from() -> None:
    """Regression: the augmented model once trained on 9.3% coverage.

    With the graph reference at 50% and the train cut at 60%, only 1 of 32
    training attack windows carried a graph score, so the extra columns were
    almost entirely missing and the "graph features do not help" result was an
    artefact of the split.
    """

    from cloudsentinel.features import build_feature_store
    from cloudsentinel.models import reference_aware_split
    from cloudsentinel.pipeline import run_pipeline
    from cloudsentinel.schema import events_to_dataframe
    from cloudsentinel.simulator import CloudSimulator, SimulationConfig

    config = SimulationConfig(
        n_identities=40,
        days=16,
        events_per_identity_per_day=12,
        attack_ratio=0.3,
        seed=91,
        campaign_start_range=(0.35, 0.95),
    )
    events, _ = run_pipeline(events_to_dataframe(CloudSimulator(config).run().events), strict=False)
    store = build_feature_store(events, graph_reference=0.3)
    split = reference_aware_split(store, 0.3, 0.65)

    train_attacks = split.train[split.train["is_attack"]]
    assert len(train_attacks) > 0
    coverage = float(train_attacks["graph_scored"].mean())
    assert coverage > 0.8, f"training attack windows with graph scores: {coverage:.1%}"
    assert float(split.test["graph_scored"].mean()) > 0.95
