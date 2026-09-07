"""Phase 9 tests: graph construction, exposure and the Graph Risk Score."""

from __future__ import annotations

import pandas as pd
import pytest

from cloudsentinel.graph import (
    GRAPH_COLUMNS,
    GraphRiskModel,
    attach_graph,
    build_graph,
    graph_stats,
    identity_exposure,
    node_id,
    observed_path,
    reference_period,
    suspicious_paths,
)
from cloudsentinel.graph.builder import IDENTITY, RESOURCE, ROLE
from cloudsentinel.pipeline import run_pipeline
from cloudsentinel.schema import events_to_dataframe
from cloudsentinel.simulator import CloudSimulator, SimulationConfig

CONFIG = SimulationConfig(
    n_identities=25, days=14, events_per_identity_per_day=16, attack_ratio=0.3, seed=31
)


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    frame, _ = run_pipeline(events_to_dataframe(CloudSimulator(CONFIG).run().events), strict=False)
    return frame


def test_graph_has_identities_resources_and_roles(events: pd.DataFrame) -> None:
    graph = build_graph(events)
    kinds = {data.get("kind") for _, data in graph.nodes(data=True)}

    assert {IDENTITY, RESOURCE, ROLE} <= kinds
    assert graph.number_of_edges() > graph.number_of_nodes()
    assert node_id(IDENTITY, events["identity_id"].iloc[0]) in graph


def test_edge_weights_count_repeated_access(events: pd.DataFrame) -> None:
    graph = build_graph(events)
    weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    assert max(weights) > 1
    assert all(weight >= 1 for weight in weights)


def test_stats_expose_centrality_communities_and_critical_assets(events: pd.DataFrame) -> None:
    graph = build_graph(events)
    stats = graph_stats(graph)

    assert stats.n_nodes == graph.number_of_nodes()
    assert len(stats.betweenness) == graph.number_of_nodes()
    assert len(set(stats.community.values())) > 1
    assert stats.critical_nodes


def test_exposure_reports_distance_and_blast_radius(events: pd.DataFrame) -> None:
    graph = build_graph(events)
    stats = graph_stats(graph)
    identity = str(events["identity_id"].iloc[0])
    exposure = identity_exposure(graph, stats, identity)

    assert exposure.blast_radius > 0
    assert 0.0 <= exposure.proximity <= 1.0
    if exposure.distance_to_critical is not None:
        assert exposure.critical_reachable > 0


def test_unknown_identity_has_no_exposure(events: pd.DataFrame) -> None:
    graph = build_graph(events)
    exposure = identity_exposure(graph, graph_stats(graph), "does_not_exist")
    assert exposure.blast_radius == 0
    assert exposure.proximity == 0.0


def test_suspicious_paths_end_on_critical_assets(events: pd.DataFrame) -> None:
    graph = build_graph(events)
    stats = graph_stats(graph)
    identity = str(events["identity_id"].iloc[0])
    paths = suspicious_paths(graph, stats, identity)

    for path in paths[:5]:
        assert path[0] == node_id(IDENTITY, identity)
        assert path[-1] in stats.critical_nodes


def test_new_edges_separate_attack_windows(events: pd.DataFrame) -> None:
    """The graph-native signal: reaching a node this identity has no edge to."""
    reference = reference_period(events, 0.5)
    model = GraphRiskModel().fit(reference)
    rows = model.score_windows(events)

    labels = (
        events.assign(window_start=events["timestamp"].dt.floor("1h"))
        .groupby(["identity_id", "window_start"], observed=True)["is_attack"]
        .max()
        .reset_index()
    )
    merged = rows.merge(labels, on=["identity_id", "window_start"], how="left")
    later = merged[merged["window_start"] > reference["timestamp"].max()]

    attack = later[later["is_attack"].astype(bool)]["graph_new_edge_ratio"]
    benign = later[~later["is_attack"].astype(bool)]["graph_new_edge_ratio"]
    assert len(attack) >= 3
    assert attack.mean() > benign.mean() + 0.1


def test_graph_reference_must_come_from_the_same_environment(events: pd.DataFrame) -> None:
    """A graph fitted elsewhere degenerates — every edge looks new."""
    other_config = SimulationConfig(
        n_identities=25, days=14, events_per_identity_per_day=16, attack_ratio=0.3, seed=77
    )
    other, _ = run_pipeline(
        events_to_dataframe(CloudSimulator(other_config).run().events), strict=False
    )
    foreign = GraphRiskModel().fit(other).score_windows(events)
    native = GraphRiskModel().fit(reference_period(events, 0.5)).score_windows(events)

    assert foreign["graph_new_edge_ratio"].mean() > 0.9
    assert native["graph_new_edge_ratio"].mean() < foreign["graph_new_edge_ratio"].mean()


def test_weight_calibration_drops_components_without_variance(events: pd.DataFrame) -> None:
    model = GraphRiskModel().fit(reference_period(events, 0.5))
    assert sum(model.effective_weights.values()) == pytest.approx(1.0)
    for component in model.inactive_components:
        assert component not in model.effective_weights


def test_attach_graph_adds_bounded_columns(events: pd.DataFrame) -> None:
    from cloudsentinel.features import build_feature_store

    store = build_feature_store(events, with_temporal=False, with_graph=False)
    out, model = attach_graph(store, reference_period(events, 0.5), events)

    assert set(GRAPH_COLUMNS) <= set(out.columns)
    assert len(out) == len(store)
    assert model.stats.n_nodes > 0

    # Only windows outside the reference period carry a score; the rest are
    # explicitly unscored rather than silently zero.
    scored = out[out["graph_scored"]]
    assert not scored.empty
    assert scored["graph_risk"].between(0, 1).all()
    assert out.loc[~out["graph_scored"], "graph_risk"].isna().all()


def test_observed_path_is_deduplicated_and_ordered() -> None:
    result = CloudSimulator(CONFIG).run()
    attack_events = [event for event in result.events if event.is_attack]
    campaign = [
        event
        for event in attack_events
        if event.labels and event.labels.campaign_id == attack_events[0].labels.campaign_id
    ]

    steps = observed_path(campaign)
    assert steps
    assert [step["order"] for step in steps] == list(range(len(steps)))
    assert len({step["node_id"] for step in steps}) == len(steps)


def test_windows_inside_the_reference_period_are_never_scored() -> None:
    """The graph must not judge a window using edges that window created.

    Before this guard, every window inside the reference period scored exactly
    0.000 new-edge ratio — including attack windows — because their own edges
    were already in the graph.
    """

    from cloudsentinel.graph import attach_graph, reference_period
    from cloudsentinel.pipeline import run_pipeline
    from cloudsentinel.schema import events_to_dataframe
    from cloudsentinel.simulator import CloudSimulator, SimulationConfig

    config = SimulationConfig(
        n_identities=25, days=12, events_per_identity_per_day=12, attack_ratio=0.3, seed=71
    )
    events, _ = run_pipeline(events_to_dataframe(CloudSimulator(config).run().events), strict=False)

    from cloudsentinel.features.windows import aggregate_windows

    rows = aggregate_windows(events)
    reference = reference_period(events, 0.5)
    merged, _model = attach_graph(rows, reference, events)

    # A window that starts before the cutoff can still contain events after it,
    # so "inside" means strictly before the window the cutoff falls in.
    cutoff = reference["timestamp"].max()
    inside = merged[merged["window_start"] < cutoff.floor("1h")]
    outside = merged[merged["window_start"] > cutoff]

    # Inside the reference period: no score at all, and no fabricated zero.
    assert inside["graph_risk"].isna().all()
    assert not inside["graph_scored"].any()

    # Outside: scored, and the novelty signal is actually able to move.
    assert outside["graph_scored"].any()
    scored = outside[outside["graph_scored"]]
    assert scored["graph_risk"].between(0.0, 1.0).all()
    assert scored["graph_new_edge_ratio"].max() > 0.0


def test_centrality_scaling_does_not_depend_on_the_batch() -> None:
    """A window's score must not change with which other windows are scored."""

    from cloudsentinel.graph.risk import GraphRiskModel, reference_period
    from cloudsentinel.pipeline import run_pipeline
    from cloudsentinel.schema import events_to_dataframe
    from cloudsentinel.simulator import CloudSimulator, SimulationConfig

    config = SimulationConfig(
        n_identities=25, days=12, events_per_identity_per_day=12, attack_ratio=0.3, seed=72
    )
    events, _ = run_pipeline(events_to_dataframe(CloudSimulator(config).run().events), strict=False)
    reference = reference_period(events, 0.5)
    later = events[events["timestamp"] > reference["timestamp"].max()]

    model = GraphRiskModel().fit(reference)
    full = model.score_windows(later).set_index(["identity_id", "window_start"])
    # Slice on a window boundary: cutting mid-window would remove events from a
    # window and change its aggregate, which is not batch dependence of the
    # scoring function but a different input.
    boundary = later["timestamp"].quantile(0.5).floor("1h")
    half = model.score_windows(later[later["timestamp"] < boundary]).set_index(
        ["identity_id", "window_start"]
    )

    common = full.index.intersection(half.index)
    assert len(common) > 100
    difference = (full.loc[common, "graph_risk"] - half.loc[common, "graph_risk"]).abs()
    assert float(difference.max()) < 1e-9
