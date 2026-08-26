from typing import Annotated

from fastapi import APIRouter, Depends, Request

from deep_research.agents.clarifier import ClarifierAgent
from deep_research.agents.planner import PlanningAgent
from deep_research.agents.questions import QuestionsAgent
from deep_research.contracts.clarification import ClarificationDecision, ClarifierRequest
from deep_research.contracts.planning import PlannerRequest, ResearchPlan
from deep_research.contracts.questions import FollowUpQuestionSet, QuestionsRequest

router = APIRouter(prefix="/v1")


def _get_clarifier(request: Request) -> ClarifierAgent:
    return request.app.state.clarifier


def _get_planner(request: Request) -> PlanningAgent:
    return request.app.state.planner


def _get_questions_agent(request: Request) -> QuestionsAgent:
    return request.app.state.questions


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/clarifier/evaluate", response_model=ClarificationDecision)
async def evaluate_clarification(
    payload: ClarifierRequest,
    clarifier: Annotated[ClarifierAgent, Depends(_get_clarifier)],
) -> ClarificationDecision:
    return await clarifier.evaluate(payload)


@router.post("/planner/generate", response_model=ResearchPlan)
async def generate_plan(
    payload: PlannerRequest,
    planner: Annotated[PlanningAgent, Depends(_get_planner)],
) -> ResearchPlan:
    return await planner.create_plan(payload)


@router.post("/questions/generate", response_model=FollowUpQuestionSet)
async def generate_questions(
    payload: QuestionsRequest,
    questions_agent: Annotated[QuestionsAgent, Depends(_get_questions_agent)],
) -> FollowUpQuestionSet:
    return await questions_agent.generate(payload)
