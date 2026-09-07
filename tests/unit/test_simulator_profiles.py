"""Phase 3 tests: environment and identity baselines."""

from __future__ import annotations

import numpy as np
import pytest

from cloudsentinel.schema.enums import IdentityType, ResourceType, Sensitivity
from cloudsentinel.simulator import build_environment, build_identities, spec_for
from cloudsentinel.simulator.catalog import CATALOG, ROLE_ACTIONS
from cloudsentinel.simulator.normal import day_factor, hour_weights
from cloudsentinel.simulator.profiles import PERSONAS


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


def test_catalog_actions_are_unique() -> None:
    actions = [spec.action for spec in CATALOG]
    assert len(actions) == len(set(actions))


def test_every_persona_action_exists_in_catalog() -> None:
    for actions in ROLE_ACTIONS.values():
        for action in actions:
            assert spec_for(action).action == action


def test_environment_has_sensitive_resources(rng: np.random.Generator) -> None:
    env = build_environment(rng)
    assert env.of_type(ResourceType.BUCKET)
    assert env.admin_roles
    assert all(r.sensitivity >= Sensitivity.CONFIDENTIAL for r in env.sensitive())
    assert env.lookup("bucket_finance") is not None


def test_identity_population_respects_human_ratio(rng: np.random.Generator) -> None:
    env = build_environment(rng)
    profiles = build_identities(
        rng, env, n_identities=100, human_ratio=0.7, events_per_day=30, business_hours=(8, 19)
    )
    humans = [p for p in profiles if p.identity_type is IdentityType.HUMAN]
    assert len(profiles) == 100
    assert len(humans) == 70
    assert {p.identity_id for p in profiles}.__len__() == 100


def test_personas_cover_every_identity_type() -> None:
    assert set(PERSONAS.values()) == set(IdentityType)


def test_humans_and_machines_have_different_rhythms(rng: np.random.Generator) -> None:
    env = build_environment(rng)
    profiles = build_identities(
        rng, env, n_identities=40, human_ratio=0.5, events_per_day=30, business_hours=(8, 19)
    )
    human = next(p for p in profiles if p.is_human)
    machine = next(p for p in profiles if not p.is_human)

    human_night = hour_weights(human)[0:6].sum()
    machine_night = hour_weights(machine)[0:6].sum()
    assert human_night < machine_night
    assert hour_weights(human)[9] > hour_weights(human)[3]


def test_seasonality_lowers_weekends_and_lifts_month_end() -> None:
    from datetime import UTC, datetime

    saturday = datetime(2026, 3, 14, tzinfo=UTC)
    tuesday = datetime(2026, 3, 17, tzinfo=UTC)
    month_end = datetime(2026, 3, 27, tzinfo=UTC)

    assert day_factor(saturday, 0.25) < day_factor(tuesday, 0.25)
    assert day_factor(month_end, 0.25) > day_factor(tuesday, 0.25)


def test_benign_traffic_contains_rare_foreign_origins(rng: np.random.Generator) -> None:
    """Foreign country and hosting ASN must not be perfect attack separators."""
    from datetime import UTC, datetime, timedelta

    from cloudsentinel.simulator import EventFactory, generate_normal_events
    from cloudsentinel.simulator.environment import FOREIGN_COUNTRIES, HOSTING_ASNS

    env = build_environment(rng)
    profiles = build_identities(
        rng, env, n_identities=60, human_ratio=1.0, events_per_day=30, business_hours=(8, 19)
    )
    factory = EventFactory(rng)
    start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
        days=40
    )

    events = [
        event
        for profile in profiles
        for event in generate_normal_events(profile, env, factory, rng, start, 40, 0.25)
    ]
    foreign = [e for e in events if e.country in FOREIGN_COUNTRIES]
    vpn = [e for e in events if e.asn in HOSTING_ASNS]

    assert foreign, "benign traffic must occasionally come from abroad"
    assert vpn, "benign traffic must occasionally come through a VPN/hosting ASN"
    # Rare enough to stay anomalous, common enough to create real FP pressure.
    assert 0 < len(foreign) / len(events) < 0.02
    assert 0 < len(vpn) / len(events) < 0.02
