"""Phase 3 tests: every scenario produces labelled, staged, valid events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from cloudsentinel.schema.enums import AttackScenario, AttackStage, IdentityType, PrivilegeLevel
from cloudsentinel.simulator import EventFactory, build_environment, build_identities
from cloudsentinel.simulator.scenarios import SCENARIOS, ScenarioContext

PAST = datetime(2026, 3, 10, 3, 0, tzinfo=UTC)


def context_for(scenario_index: int) -> tuple[ScenarioContext, object]:
    rng = np.random.default_rng(5)
    env = build_environment(rng)
    profiles = build_identities(
        rng, env, n_identities=80, human_ratio=0.5, events_per_day=25, business_hours=(8, 19)
    )
    scenario = SCENARIOS[scenario_index]
    profile = next(p for p in profiles if p.identity_type in scenario.applies_to)
    ctx = ScenarioContext(profile, env, EventFactory(rng), rng, PAST, "camp_test")
    return ctx, scenario


@pytest.mark.parametrize("index", range(len(SCENARIOS)))
def test_scenario_generates_labelled_events(index: int) -> None:
    ctx, scenario = context_for(index)
    events = scenario.generate(ctx)  # type: ignore[attr-defined]

    assert len(events) >= 5
    assert all(event.is_attack for event in events)
    assert all(event.scenario is scenario.name for event in events)  # type: ignore[attr-defined]
    assert all(event.labels and event.labels.campaign_id == "camp_test" for event in events)
    assert all(event.labels and event.labels.attack_stage is not None for event in events)


def test_all_ten_scenarios_are_registered() -> None:
    assert len(SCENARIOS) == 10
    assert {s.name for s in SCENARIOS} == set(AttackScenario)


def test_credential_compromise_uses_unfamiliar_origin() -> None:
    ctx, scenario = context_for(0)
    events = scenario.generate(ctx)  # type: ignore[attr-defined]
    known_ips = {str(ip) for ip in ctx.profile.ip_pool}

    assert all(str(event.source_ip) not in known_ips for event in events)
    assert all(event.country != ctx.profile.home_country for event in events)
    assert any(not event.success for event in events)


def test_account_takeover_removes_mfa_then_escalates() -> None:
    ctx, scenario = context_for(1)
    actions = [event.action for event in scenario.generate(ctx)]  # type: ignore[attr-defined]
    assert actions.index("DeactivateMFADevice") < actions.index("AttachRolePolicy")


def test_privilege_escalation_reaches_admin() -> None:
    ctx, scenario = context_for(2)
    events = scenario.generate(ctx)  # type: ignore[attr-defined]
    admin_ids = {role.resource_id for role in ctx.environment.admin_roles}
    escalation = [
        e for e in events if e.labels and e.labels.attack_stage is AttackStage.PRIVILEGE_ESCALATION
    ]

    assert escalation
    assert any(event.resource_id in admin_ids for event in escalation)
    assert max(event.privilege_level for event in events) >= PrivilegeLevel.ADMIN


def test_exfiltration_volume_ramps_up() -> None:
    ctx, scenario = context_for(3)
    events = [e for e in scenario.generate(ctx) if e.action == "GetObject"]  # type: ignore[attr-defined]
    first_third = sum(e.bytes_out for e in events[:8])
    last_third = sum(e.bytes_out for e in events[-8:])

    # A gradual ramp, not a spike: the temporal layer should be what catches it.
    assert last_third > first_third * 3
    assert events[-1].objects_accessed > events[0].objects_accessed * 5
    assert any(e.labels and e.labels.attack_stage is AttackStage.EXFILTRATION for e in events)


def test_cryptomining_runs_in_a_cold_region_and_persists() -> None:
    ctx, scenario = context_for(5)
    events = scenario.generate(ctx)  # type: ignore[attr-defined]
    regions = {event.region for event in events}

    assert ctx.profile.region not in regions
    span = max(e.timestamp for e in events) - min(e.timestamp for e in events)
    assert span > timedelta(hours=3)


def test_suspicious_service_account_uses_unseen_apis() -> None:
    ctx, scenario = context_for(6)
    events = scenario.generate(ctx)  # type: ignore[attr-defined]
    baseline = set(ctx.profile.actions)
    assert any(event.action not in baseline for event in events)


def test_ai_agent_scenario_targets_agents_only() -> None:
    ctx, scenario = context_for(9)
    assert ctx.profile.identity_type is IdentityType.AI_AGENT
    events = scenario.generate(ctx)  # type: ignore[attr-defined]
    assert any(event.action == "agent:read_secret" for event in events)


def test_scenarios_reuse_the_normal_action_vocabulary() -> None:
    """Attacks must not be separable by action name alone."""
    from cloudsentinel.simulator.catalog import BY_NAME

    for index in range(len(SCENARIOS)):
        ctx, scenario = context_for(index)
        for event in scenario.generate(ctx):  # type: ignore[attr-defined]
            assert event.action in BY_NAME
