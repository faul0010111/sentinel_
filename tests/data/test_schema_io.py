"""Phase 2 tests: storage round-trips and the data-quality gate."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cloudsentinel.schema import (
    AttackScenario,
    CloudEvent,
    EventLabels,
    EventType,
    IdentityType,
    ResourceType,
    events_to_dataframe,
    read_events_jsonl,
    read_events_parquet,
    validate_records,
    write_events_jsonl,
    write_events_parquet,
)


def sample_events() -> list[CloudEvent]:
    common = {
        "event_type": EventType.API_CALL,
        "identity_type": IdentityType.SERVICE_ACCOUNT,
        "service": "s3",
        "action": "ListObjects",
        "resource_type": ResourceType.BUCKET,
        "source_ip": "203.0.113.7",
        "country": "BR",
    }
    return [
        CloudEvent(
            timestamp=datetime(2026, 3, 14, 9, 0, tzinfo=UTC),
            identity_id="svc_backup",
            resource_id="bucket_a",
            bytes_out=1024,
            **common,  # type: ignore[arg-type]
        ),
        CloudEvent(
            timestamp=datetime(2026, 3, 14, 8, 0, tzinfo=UTC),
            identity_id="svc_backup",
            resource_id="bucket_b",
            bytes_out=999_999,
            labels=EventLabels(is_attack=True, scenario=AttackScenario.DATA_EXFILTRATION),
            **common,  # type: ignore[arg-type]
        ),
    ]


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = write_events_jsonl(sample_events(), tmp_path / "events.jsonl")
    events, report = read_events_jsonl(path)
    assert report.invalid == 0
    assert report.valid_ratio == 1.0
    assert {event.resource_id for event in events} == {"bucket_a", "bucket_b"}


def test_parquet_round_trip_preserves_labels(tmp_path: Path) -> None:
    path = write_events_parquet(sample_events(), tmp_path / "events.parquet")
    events, report = read_events_parquet(path)
    assert report.invalid == 0
    attacks = [event for event in events if event.is_attack]
    assert len(attacks) == 1
    assert attacks[0].scenario is AttackScenario.DATA_EXFILTRATION
    assert attacks[0].resource_id == "bucket_b"


def test_dataframe_is_sorted_and_flattened() -> None:
    frame = events_to_dataframe(sample_events())
    assert list(frame["resource_id"]) == ["bucket_b", "bucket_a"]
    assert frame["is_attack"].tolist() == [True, False]
    assert "labels" not in frame.columns


def test_bad_records_are_quarantined_not_raised() -> None:
    good = sample_events()[0].model_dump(mode="json")
    bad = {**good, "country": "BRAZIL", "bytes_out": -5}
    events, report = validate_records([good, bad, {"identity_id": "x"}])
    assert report.total == 3
    assert report.valid == 1
    assert report.invalid == 2
    assert len(events) == 1
    fields = {error.field for error in report.errors}
    assert "country" in fields
    assert "bytes_out" in fields


def test_error_cap_is_respected() -> None:
    records = [{"identity_id": "x"} for _ in range(50)]
    _, report = validate_records(records, max_errors=5)
    assert report.invalid == 50
    assert len(report.errors) == 5
