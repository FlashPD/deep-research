from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from deep_research.agents.clarifier import ClarifierAgent
from deep_research.agents.planner import PlanningAgent
from deep_research.agents.questions import QuestionsAgent
from deep_research.agents.report import ReportGenerationAgent
from deep_research.agents.researcher import ResearchAgent
from deep_research.agents.reviewer import EvidenceReviewer
from deep_research.api.routes import router
from deep_research.auth.cognito import (
    AuthenticationError,
    RequestAuthenticator,
    build_authenticator,
)
from deep_research.models.gateway import ModelGateway, StructuredModelGateway
from deep_research.persistence.dynamodb import DynamoDBRunRepository
from deep_research.persistence.memory import InMemoryRunRepository
from deep_research.persistence.runs import (
    ConcurrencyConflictError,
    RunNotFoundError,
    RunRepository,
)
from deep_research.services.runs import (
    RunCommandConflictError,
    RunControlService,
)
from deep_research.settings import get_settings
from deep_research.tools.research import ResearchAdapter, UnconfiguredResearchAdapter


def create_app(
    gateway: StructuredModelGateway | None = None,
    research_adapter: ResearchAdapter | None = None,
    run_repository: RunRepository | None = None,
    authenticator: RequestAuthenticator | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        configured_gateway = gateway or ModelGateway(settings.load_models())
        app.state.clarifier = ClarifierAgent(configured_gateway)
        app.state.planner = PlanningAgent(configured_gateway)
        app.state.questions = QuestionsAgent(configured_gateway)
        app.state.reviewer = EvidenceReviewer(configured_gateway)
        app.state.report = ReportGenerationAgent(configured_gateway)
        app.state.researcher = ResearchAgent(
            configured_gateway, research_adapter or UnconfiguredResearchAdapter()
        )
        configured_repository = run_repository or _build_run_repository(settings)
        app.state.run_control = RunControlService(
            configured_repository,
            retention_days=settings.run_retention_days,
        )
        app.state.authenticator = authenticator or build_authenticator(settings)
        yield

    app = FastAPI(title="Deep Research API", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate_request(request, call_next):
        if request.url.path == "/v1/health" or not request.url.path.startswith("/v1/"):
            return await call_next(request)
        try:
            request.state.principal = await request.app.state.authenticator.authenticate(
                request.headers.get("Authorization")
            )
        except AuthenticationError as exc:
            headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": str(exc)},
                headers=headers,
            )
        return await call_next(request)

    app.add_exception_handler(
        RunNotFoundError,
        lambda _request, exc: JSONResponse(status_code=404, content={"detail": str(exc)}),
    )
    for error_type in (
        ConcurrencyConflictError,
        RunCommandConflictError,
    ):
        app.add_exception_handler(
            error_type,
            lambda _request, exc: JSONResponse(status_code=409, content={"detail": str(exc)}),
        )
    app.include_router(router)
    return app


def _build_run_repository(settings) -> RunRepository:
    if not settings.dynamodb_runs_table:
        return InMemoryRunRepository()
    import boto3

    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.dynamodb_runs_table
    )
    return DynamoDBRunRepository(table)


app = create_app()
