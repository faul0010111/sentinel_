"""Phase 3 tests: full runs, determinism and persisted ground truth."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cloudsentinel.schema import read_events_parquet
from cloudsentinel.simulator import CloudSimulator, SimulationConfig
from cloudsentinel.simulator.writer import manifest_path_for, read_manifest, write_simulation

SMALL = SimulationConfig(
    n_identities=20, days=8, events_per_identity_per_day=15, attack_ratio=0.25, seed=99
)


def test_run_produces_labelled_dataset() -> None:
    result = CloudSimulator(SMALL).run()
    manifest = result.manifest

    assert manifest.n_events > 500
    assert manifest.campaigns
    assert manifest.n_attack_events == sum(c.n_events for c in manifest.campaigns)
    assert 0 < manifest.attack_ratio < 0.5


def test_same_seed_reproduces_the_dataset() -> None:
    first = CloudSimulator(SMALL).run().events
    second = CloudSimulator(SMALL).run().events

    assert len(first) == len(second)
    assert [e.event_id for e in first] == [e.event_id for e in second]
    assert [e.bytes_out for e in first] == [e.bytes_out for e in second]


def test_different_seed_changes_the_dataset() -> None:
    other = replace(SMALL, seed=1234)
    assert (
        CloudSimulator(SMALL).run().manifest.n_events
        != CloudSimulator(other).run().manifest.n_events
    )


def test_events_are_chronological_and_mostly_benign() -> None:
    result = CloudSimulator(SMALL).run()
    stamps = [event.timestamp for event in result.events]
    assert stamps == sorted(stamps)
    assert result.manifest.attack_ratio < 0.2


def test_only_campaign_identities_carry_labels() -> None:
    result = CloudSimulator(SMALL).run()
    compromised = {campaign.identity_id for campaign in result.manifest.campaigns}
    labelled = {event.identity_id for event in result.events if event.is_attack}
    assert labelled == compromised


def test_campaigns_start_after_a_clean_baseline() -> None:
    simulator = CloudSimulator(SMALL)
    result = simulator.run()
    for campaign in result.manifest.campaigns:
        elapsed = (campaign.start - simulator.start).days
        assert elapsed >= SMALL.days * 0.5


def test_write_simulation_persists_events_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "events.parquet"
    manifest = write_simulation(CloudSimulator(SMALL), output)

    assert output.exists()
    events, report = read_events_parquet(output)
    assert report.invalid == 0
    assert len(events) == manifest.n_events

    reloaded = read_manifest(manifest_path_for(output))
    assert reloaded.n_events == manifest.n_events
    assert reloaded.per_scenario() == manifest.per_scenario()


def test_jsonl_output_is_supported(tmp_path: Path) -> None:
    output = tmp_path / "events.jsonl"
    manifest = write_simulation(CloudSimulator(SMALL), output)
    assert output.exists()
    assert manifest.n_events == sum(1 for _ in output.open())


def test_every_scenario_is_covered_when_budget_allows() -> None:
    """Per-scenario recall is only measurable if each scenario is present."""
    from cloudsentinel.schema.enums import AttackScenario

    config = SimulationConfig(
        n_identities=200, days=6, events_per_identity_per_day=6, attack_ratio=0.1, seed=4
    )
    simulator = CloudSimulator(config)
    assert set(simulator.compromised.values()) == set(AttackScenario)


def test_small_budget_still_produces_at_least_one_campaign() -> None:
    config = SimulationConfig(
        n_identities=10, days=4, events_per_identity_per_day=5, attack_ratio=0.01, seed=8
    )
    assert len(CloudSimulator(config).compromised) == 1


def test_campaigns_never_leave_the_simulated_window() -> None:
    """Regression: a short run put campaigns past the window, i.e. in the future.

    The window ends at the previous midnight, so an offset of 0.95 days plus up
    to 24h of jitter on a 6-day run produced timestamps the schema rejects. It
    only failed when the suite ran late in the day.
    """
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    for days in (3, 6, 14):
        config = SimulationConfig(
            n_identities=8,
            days=days,
            events_per_identity_per_day=6,
            attack_ratio=0.5,
            seed=13,
            campaign_start_range=(0.66, 0.95),
        )
        simulator = CloudSimulator(config)
        result = simulator.run()

        latest = max(event.timestamp for event in result.events)
        assert latest <= now
        for campaign in result.manifest.campaigns:
            assert campaign.end <= now
            assert campaign.start >= simulator.start
