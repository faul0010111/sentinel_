"""Phase 2 tests: the event contract accepts good data and rejects bad data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from cloudsentinel.schema import (
    LABEL_FIELDS,
    SCHEMA_VERSION,
    AttackScenario,
    CloudEvent,
    EventBatch,
    EventLabels,
    EventType,
    IdentityType,
    PrivilegeLevel,
    ResourceType,
)

BASE = {
    "timestamp": datetime(2026, 3, 14, 9, 12, 44, tzinfo=UTC),
    "event_type": EventType.IAM,
    "identity_id": "user_001",
    "identity_type": IdentityType.HUMAN,
    "source_ip": "10.0.0.1",
    "country": "BR",
    "asn": "AS28573",
    "service": "iam",
    "action": "AssumeRole",
    "resource_id": "resource_001",
    "resource_type": ResourceType.ROLE,
    "privilege_level": PrivilegeLevel.ADMIN,
}


def make_event(**overrides: object) -> CloudEvent:
    return CloudEvent(**{**BASE, **overrides})  # type: ignore[arg-type]


def test_minimal_event_is_valid() -> None:
    event = make_event()
    assert event.schema_version == SCHEMA_VERSION
    assert event.privilege_level == PrivilegeLevel.ADMIN
    assert not event.is_attack


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        make_event(sourceIPAddress="10.0.0.2")


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_event(timestamp=datetime(2026, 3, 14, 9, 0, 0))


def test_future_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError, match="future"):
        make_event(timestamp=datetime.now(UTC) + timedelta(hours=2))


def test_timestamp_is_normalized_to_utc() -> None:
    stamp = datetime(2026, 3, 14, 9, 0, tzinfo=UTC).astimezone()
    assert make_event(timestamp=stamp).timestamp.tzinfo is UTC


@pytest.mark.parametrize("value", ["br", " BR ", "BR"])
def test_country_is_normalized(value: str) -> None:
    assert make_event(country=value).country == "BR"


def test_invalid_country_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_event(country="BRA")


@pytest.mark.parametrize(("value", "expected"), [("28573", "AS28573"), ("as28573", "AS28573")])
def test_asn_is_normalized(value: str, expected: str) -> None:
    assert make_event(asn=value).asn == expected


def test_invalid_ip_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_event(source_ip="999.1.1.1")


def test_negative_bytes_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_event(bytes_out=-1)


def test_success_with_error_code_is_rejected() -> None:
    with pytest.raises(ValidationError, match="error_code"):
        make_event(success=True, error_code="AccessDenied")


def test_data_access_requires_objects() -> None:
    with pytest.raises(ValidationError, match="objects_accessed"):
        make_event(event_type=EventType.DATA_ACCESS, objects_accessed=0)


def test_agent_action_cannot_be_human() -> None:
    with pytest.raises(ValidationError, match="agent tool calls"):
        make_event(action="agent:search_documents")


def test_agent_action_allowed_for_ai_agent() -> None:
    event = make_event(identity_type=IdentityType.AI_AGENT, action="agent:search_documents")
    assert event.identity_type is IdentityType.AI_AGENT


def test_labels_require_a_scenario() -> None:
    with pytest.raises(ValidationError, match="must name a scenario"):
        EventLabels(is_attack=True)


def test_scenario_requires_attack_flag() -> None:
    with pytest.raises(ValidationError, match="only be set on events labelled as attack"):
        EventLabels(is_attack=False, scenario=AttackScenario.LATERAL_MOVEMENT)


def test_feature_payload_drops_labels_and_raw() -> None:
    event = make_event(
        raw={"eventSource": "iam.amazonaws.com"},
        labels=EventLabels(is_attack=True, scenario=AttackScenario.PRIVILEGE_ESCALATION),
    )
    payload = event.feature_payload()
    assert event.is_attack is True
    assert LABEL_FIELDS.isdisjoint(payload)
    assert "raw" not in payload
    assert payload["identity_id"] == "user_001"


def test_batch_time_range() -> None:
    early = make_event(timestamp=datetime(2026, 3, 14, 8, 0, tzinfo=UTC))
    late = make_event(timestamp=datetime(2026, 3, 14, 10, 0, tzinfo=UTC))
    batch = EventBatch(events=[late, early], source="test")
    assert len(batch) == 2
    assert batch.time_range == (early.timestamp, late.timestamp)


def test_json_round_trip_is_lossless() -> None:
    event = make_event(labels=EventLabels(is_attack=True, scenario=AttackScenario.CRYPTOMINING))
    assert CloudEvent.model_validate_json(event.model_dump_json()) == event
