from fastapi.testclient import TestClient
from pydantic import SecretStr

from deep_research.contracts.research import UploadSearchOperation
from deep_research.dev_app import create_dev_app
from deep_research.jobs.memory import InMemoryJobDispatcher
from deep_research.settings import AppSettings
from deep_research.tools.live import build_live_research_services
from deep_research.tools.research import ResearchServiceConfigurationError


def test_development_runtime_shares_control_and_dispatcher() -> None:
    settings = AppSettings(
        model_provider="openai",
        openai_api_key=SecretStr("test-openai-key"),
        tavily_api_key=SecretStr("test-tavily-key"),
    )
    app = create_dev_app(settings)

    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 200
        assert app.state.background_worker._control is app.state.run_control
        assert app.state.background_worker._dispatcher is app.state.job_dispatcher
        assert isinstance(app.state.job_dispatcher, InMemoryJobDispatcher)


async def test_public_web_services_do_not_construct_upload_infrastructure(monkeypatch) -> None:
    def unexpected_upload_construction(**_kwargs):
        raise AssertionError("OpenSearch must remain lazy in public-web mode")

    monkeypatch.setattr(
        "deep_research.tools.live.OpenSearchUploadIndex.from_url",
        unexpected_upload_construction,
    )
    settings = AppSettings(
        tavily_api_key=SecretStr("test-tavily-key"), uploads_enabled=False
    )

    services = build_live_research_services(settings, tenant_id="tenant", run_id="run")

    try:
        await services.research_adapter.search_uploads(
            UploadSearchOperation(query="private evidence", research_question_ids=["question"])
        )
    except ResearchServiceConfigurationError as exc:
        assert "UPLOADS_ENABLED" in str(exc)
    else:
        raise AssertionError("an upload plan must fail with a typed configuration error")
