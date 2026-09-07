"""The streaming loop: window closing, late events and source correctness."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from cloudsentinel.schema.enums import EventType, IdentityType
from cloudsentinel.schema.events import CloudEvent
from cloudsentinel.simulator import CloudSimulator, SimulationConfig
from cloudsentinel.simulator.writer import write_simulation
from cloudsentinel.streaming import FileSource, KafkaSource, WindowBuffer, consume

BASE = datetime(2026, 3, 14, 9, 0, tzinfo=UTC)


def event(minutes: int, identity: str = "svc_a") -> CloudEvent:
    return CloudEvent(
        timestamp=BASE + timedelta(minutes=minutes),
        identity_id=identity,
        identity_type=IdentityType.SERVICE_ACCOUNT,
        event_type=EventType.API_CALL,
        service="s3",
        action="ListBuckets",
    )


def test_a_window_is_released_only_once_it_closes() -> None:
    """Scoring a partial window would disagree with the batch pipeline."""
    buffer = WindowBuffer(window="1h")

    assert buffer.add(event(0)) == []
    assert buffer.add(event(30)) == []

    released = buffer.add(event(70))  # first event of the next hour
    assert len(released) == 1
    start, events = released[0]
    assert start == pd.Timestamp(BASE).floor("1h")
    assert len(events) == 2


def test_late_events_are_dropped_not_silently_reopened() -> None:
    buffer = WindowBuffer(window="1h")
    buffer.add(event(0))
    buffer.add(event(70))  # closes the 09:00 window

    assert buffer.add(event(10)) == []
    assert buffer.dropped_late == 1


def test_flush_releases_the_tail() -> None:
    buffer = WindowBuffer(window="1h")
    buffer.add(event(0))
    buffer.add(event(5))
    assert len(buffer.flush()) == 1
    assert buffer.pending == {}


def test_consume_scores_every_closed_window(tmp_path: Path) -> None:
    config = SimulationConfig(
        n_identities=10, days=4, events_per_identity_per_day=10, attack_ratio=0.3, seed=6
    )
    path = tmp_path / "events.parquet"
    write_simulation(CloudSimulator(config), path)

    seen: list[int] = []

    def score(events: list[CloudEvent]) -> float:
        seen.append(len(events))
        return min(len(events) / 20.0, 1.0)

    stats = consume(FileSource(path), score, threshold=0.5)

    assert stats.events > 100
    assert stats.windows_scored == len(seen)
    assert sum(seen) == stats.events
    assert stats.dropped_late == 0


def test_parquet_replay_reconstructs_valid_events(tmp_path: Path) -> None:
    """Regression: reading parquet row by row rejected every event.

    The writer flattens labels into columns the event schema forbids, so the
    naive path validated nothing and the loop reported zero events while
    happily claiming success.
    """
    config = SimulationConfig(
        n_identities=6, days=3, events_per_identity_per_day=8, attack_ratio=0.4, seed=8
    )
    path = tmp_path / "events.parquet"
    manifest = write_simulation(CloudSimulator(config), path)

    events = list(FileSource(path).stream())
    assert len(events) == manifest.n_events
    assert all(isinstance(item, CloudEvent) for item in events)
    assert any(item.is_attack for item in events)


def test_jsonl_replay_works_too(tmp_path: Path) -> None:
    config = SimulationConfig(
        n_identities=5, days=2, events_per_identity_per_day=6, attack_ratio=0.4, seed=9
    )
    path = tmp_path / "events.jsonl"
    manifest = write_simulation(CloudSimulator(config), path)

    assert len(list(FileSource(path).stream())) == manifest.n_events


def test_kafka_source_is_declared_but_not_pretended() -> None:
    source = KafkaSource(topic="events", bootstrap_servers="localhost:9092")
    assert source.group_id == "cloudsentinel"
    pytest.importorskip("confluent_kafka", reason="only meaningful without the client installed")
