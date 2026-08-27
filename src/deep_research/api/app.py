from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from deep_research.agents.clarifier import ClarifierAgent
from deep_research.agents.planner import PlanningAgent
from deep_research.agents.questions import QuestionsAgent
from deep_research.agents.report import ReportGenerationAgent
from deep_research.agents.reviewer import EvidenceReviewer
from deep_research.api.routes import router
from deep_research.models.gateway import ModelGateway, StructuredModelGateway
from deep_research.settings import get_settings


def create_app(gateway: StructuredModelGateway | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configured_gateway = gateway or ModelGateway(get_settings().load_models())
        app.state.clarifier = ClarifierAgent(configured_gateway)
        app.state.planner = PlanningAgent(configured_gateway)
        app.state.questions = QuestionsAgent(configured_gateway)
        app.state.reviewer = EvidenceReviewer(configured_gateway)
        app.state.report = ReportGenerationAgent(configured_gateway)
        yield

    app = FastAPI(title="Deep Research API", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
