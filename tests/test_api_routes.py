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
    assert "/v1/runs/{run_id}/report" in paths


def test_production_api_schema_excludes_long_running_agent_execution() -> None:
    paths = create_app().openapi()["paths"]

    assert "/v1/researcher/research" not in paths
    assert "/v1/reviewer/review" not in paths
    assert "/v1/runs/{run_id}/start" in paths


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
        report_not_ready = client.get(f"/v1/runs/{created['run_id']}/report")
        assert report_not_ready.status_code == 409

        no_pending_questions = client.post(
            f"/v1/runs/{created['run_id']}/clarifications",
            headers={"Idempotency-Key": "answer-round-key"},
            json={
                "round_number": 1,
                "answers": [{"question_id": "audience", "value": "Engineers"}],
            },
        )
        assert no_pending_questions.status_code == 409
        assert "stale round" in no_pending_questions.json()["detail"]

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


def test_completed_report_endpoint_returns_markdown_for_authenticated_owner() -> None:
    class ReportControl:
        principal = None

        async def get_report(self, principal, run_id):
            self.principal = principal
            assert run_id == "0" * 32
            return "# Completed report\n"

    control = ReportControl()
    app = create_app(
        run_control=control,
        authenticator=DevelopmentAuthenticator(subject="report-owner"),
    )
    with TestClient(app) as client:
        response = client.get(f"/v1/runs/{'0' * 32}/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text == "# Completed report\n"
    assert control.principal.subject == "report-owner"
