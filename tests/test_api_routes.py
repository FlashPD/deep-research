from fastapi.testclient import TestClient

from deep_research.api.app import create_app
from deep_research.auth.cognito import AuthenticationError, DevelopmentAuthenticator
from deep_research.persistence.memory import InMemoryRunRepository
from tests.conftest import FakeGateway


def test_review_and_report_routes_are_exposed() -> None:
    paths = create_app(FakeGateway()).openapi()["paths"]

    assert "/v1/reviewer/review" in paths
    assert "/v1/report/generate" in paths
    assert "/v1/researcher/research" in paths
    assert "/v1/runs" in paths
    assert "/v1/runs/{run_id}/plan/approve" in paths
    assert "/v1/runs/{run_id}/events" in paths


def test_run_api_creates_reads_starts_and_replays_events() -> None:
    app = create_app(
        FakeGateway(),
        run_repository=InMemoryRunRepository(),
        authenticator=DevelopmentAuthenticator(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/runs",
            headers={"Idempotency-Key": "create-run-key"},
            json={"topic": "Grid storage", "depth": "quick"},
        )
        assert response.status_code == 201
        created = response.json()

        replay = client.post(
            "/v1/runs",
            headers={"Idempotency-Key": "create-run-key"},
            json={"topic": "Grid storage", "depth": "quick"},
        )
        assert replay.status_code == 201
        assert replay.json()["run_id"] == created["run_id"]

        started = client.post(
            f"/v1/runs/{created['run_id']}/start",
            headers={"Idempotency-Key": "start-run-key"},
        )
        assert started.status_code == 200
        assert started.json()["state"] == "CLARIFYING"

        events = client.get(f"/v1/runs/{created['run_id']}/events?after=1")
        assert events.status_code == 200
        assert [event["event_type"] for event in events.json()["events"]] == ["run.started"]


class RejectingAuthenticator:
    async def authenticate(self, _authorization):
        raise AuthenticationError("missing bearer token")


def test_protected_routes_require_authentication() -> None:
    app = create_app(FakeGateway(), authenticator=RejectingAuthenticator())
    with TestClient(app) as client:
        response = client.get("/v1/runs/00000000000000000000000000000000")
        health = client.get("/v1/health")

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert health.status_code == 200
