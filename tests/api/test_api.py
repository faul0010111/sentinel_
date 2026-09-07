"""Phase 12 tests: the HTTP contract, including the not-ready path."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cloudsentinel.api.service import DetectionService, reset_service
from cloudsentinel.features import FEATURE_COLUMNS, write_feature_store
from cloudsentinel.pipeline import run_pipeline
from cloudsentinel.schema import events_to_dataframe
from cloudsentinel.simulator import CloudSimulator, SimulationConfig


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    from cloudsentinel.api.main import app
    from cloudsentinel.features import build_feature_store

    config = SimulationConfig(
        n_identities=30,
        days=14,
        events_per_identity_per_day=12,
        attack_ratio=0.3,
        seed=5,
        campaign_start_range=(0.35, 0.95),
    )
    events, _ = run_pipeline(events_to_dataframe(CloudSimulator(config).run().events), strict=False)
    store = build_feature_store(events, graph_reference=0.3)

    path = tmp_path_factory.mktemp("api") / "store.parquet"
    write_feature_store(store, path)

    service = DetectionService(store_path=Path(path))
    service.load()
    reset_service(service)
    with TestClient(app) as test_client:
        yield test_client
    reset_service(None)


def test_health_reports_readiness(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["windows"] > 0


def test_metrics_is_prometheus_text(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "cloudsentinel_ready 1" in response.text
    assert "# TYPE cloudsentinel_alerts gauge" in response.text


def test_alerts_are_ranked_and_paginated(client: TestClient) -> None:
    alerts = client.get("/alerts", params={"limit": 5}).json()
    assert alerts
    scores = [alert["risk_score"] for alert in alerts]
    assert scores == sorted(scores, reverse=True)

    second = client.get("/alerts", params={"limit": 2, "offset": 1}).json()
    assert second[0]["alert_id"] == alerts[1]["alert_id"]


def test_alert_detail_and_404(client: TestClient) -> None:
    alert_id = client.get("/alerts", params={"limit": 1}).json()[0]["alert_id"]
    assert client.get(f"/alerts/{alert_id}").json()["alert_id"] == alert_id
    assert client.get("/alerts/00000000-0000-4000-8000-000000000000").status_code == 404


def test_alert_severity_matches_its_score(client: TestClient) -> None:
    from cloudsentinel.settings import get_settings

    for alert in client.get("/alerts", params={"limit": 20}).json():
        expected = get_settings().risk.severity_for(alert["risk_score"])
        assert alert["severity"] == expected


def test_severity_filter(client: TestClient) -> None:
    filtered = client.get("/alerts", params={"severity": "CRITICAL", "limit": 50}).json()
    assert all(alert["severity"] == "CRITICAL" for alert in filtered)
    assert client.get("/alerts", params={"severity": "NOPE"}).status_code == 422


def test_identities_and_identity_risk(client: TestClient) -> None:
    identities = client.get("/identities", params={"limit": 5}).json()
    assert identities
    assert identities[0]["max_score"] >= identities[-1]["max_score"]

    risk = client.get(f"/identities/{identities[0]['identity_id']}/risk").json()
    assert risk["identity"] == identities[0]["identity_id"]
    assert 0 <= risk["risk"]["score"] <= 100
    assert 0 <= risk["risk"]["confidence"] <= 1

    assert client.get("/identities/does-not-exist/risk").status_code == 404


def test_explanation_is_available_for_every_alert(client: TestClient) -> None:
    """An alert without an explanation is the worst failure of this layer."""
    alerts = client.get("/alerts", params={"limit": 25}).json()
    for alert in alerts:
        body = client.get(f"/explanations/{alert['alert_id']}").json()
        assert "WHY WAS THIS ALERT GENERATED?" in body["narrative"]
        assert body["method"]


def test_predict_scores_rows_and_reports_missing_features(client: TestClient) -> None:
    body = client.post("/predict", json={"rows": [{"event_count": 500.0, "bytes_out": 9e7}]}).json()

    assert 0.0 <= body["scores"][0] <= 1.0
    assert isinstance(body["alerts"][0], bool)
    assert len(body["missing_features"]) == len(FEATURE_COLUMNS) - 2


def test_predict_rejects_an_empty_batch(client: TestClient) -> None:
    assert client.post("/predict", json={"rows": []}).status_code == 422


def test_events_validates_and_counts(client: TestClient) -> None:
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "identity_id": "user_001",
        "identity_type": "human",
        "event_type": "api_call",
        "service": "s3",
        "action": "ListBuckets",
    }
    body = client.post("/events", json={"events": [event], "source": "test"}).json()
    assert body["accepted"] == 1
    assert body["rejected"] == 0
    assert "cloudsentinel_events_received_total 1" in client.get("/metrics").text


def test_events_rejects_a_malformed_batch(client: TestClient) -> None:
    assert client.post("/events", json={"events": [{"identity_id": "x"}]}).status_code == 422


def test_models_lists_the_registry_and_importance(client: TestClient) -> None:
    body = client.get("/models").json()
    assert "random_forest" in body["available"]
    assert body["active"] == "random_forest"
    assert body["attribution"].startswith("shap")
    assert body["feature_importance"][0]["importance"] > 0


def test_attack_paths_returns_only_reconstructed_paths(client: TestClient) -> None:
    for entry in client.get("/attack-paths").json():
        assert entry["path"]


def test_missing_store_reports_503_not_a_crash(tmp_path: Path) -> None:
    from cloudsentinel.api.main import app

    service = DetectionService(store_path=tmp_path / "absent.parquet")
    service.load()
    reset_service(service)
    try:
        with TestClient(app) as unready:
            health = unready.get("/health").json()
            assert health["status"] == "ok"
            assert health["ready"] is False
            assert "not found" in health["detail"]
            # Partial state must not leak into a not-ready service.
            assert health["windows"] == 0
            assert health["alerts"] == 0
            assert unready.get("/alerts").status_code == 503
            prediction = unready.post("/predict", json={"rows": [{"event_count": 1.0}]})
            assert prediction.status_code == 503
    finally:
        reset_service(None)
