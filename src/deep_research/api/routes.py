from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import PlainTextResponse

from deep_research.agents.clarifier import ClarifierAgent
from deep_research.agents.planner import PlanningAgent
from deep_research.agents.questions import QuestionsAgent
from deep_research.agents.report import ReportGenerationAgent
from deep_research.agents.researcher import ResearchAgent
from deep_research.agents.reviewer import EvidenceReviewer
from deep_research.contracts.clarification import ClarificationDecision, ClarifierRequest
from deep_research.contracts.evidence import ReviewerRequest, ReviewResult
from deep_research.contracts.jobs import JobPhase
from deep_research.contracts.planning import PlannerRequest, ResearchPlan
from deep_research.contracts.questions import FollowUpQuestionSet, QuestionsRequest
from deep_research.contracts.reporting import ReportArtifact, ReportRequest
from deep_research.contracts.research import ResearchRequest, ResearchResult
from deep_research.contracts.runs import (
    ClarificationAnswerRequest,
    CreateRunRequest,
    PlanApprovalRequest,
    Principal,
    ResearchRun,
    RunEventPage,
)
from deep_research.jobs.base import JobDispatcher
from deep_research.services.runs import RunControlService
from deep_research.worker import make_phase_job

router = APIRouter(prefix="/v1")
agent_router = APIRouter(prefix="/v1")
IdempotencyKey = Annotated[
    str,
    Header(alias="Idempotency-Key", min_length=8, max_length=200),
]


def _get_clarifier(request: Request) -> ClarifierAgent:
    return request.app.state.clarifier


def _get_planner(request: Request) -> PlanningAgent:
    return request.app.state.planner


def _get_questions_agent(request: Request) -> QuestionsAgent:
    return request.app.state.questions


def _get_reviewer(request: Request) -> EvidenceReviewer:
    return request.app.state.reviewer


def _get_report_agent(request: Request) -> ReportGenerationAgent:
    return request.app.state.report


def _get_research_agent(request: Request) -> ResearchAgent:
    return request.app.state.researcher


def _get_run_control(request: Request) -> RunControlService:
    return request.app.state.run_control


def _get_job_dispatcher(request: Request) -> JobDispatcher:
    return request.app.state.job_dispatcher


def _get_principal(request: Request) -> Principal:
    return request.state.principal


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@agent_router.post("/clarifier/evaluate", response_model=ClarificationDecision)
async def evaluate_clarification(
    payload: ClarifierRequest,
    clarifier: Annotated[ClarifierAgent, Depends(_get_clarifier)],
) -> ClarificationDecision:
    return await clarifier.evaluate(payload)


@agent_router.post("/planner/generate", response_model=ResearchPlan)
async def generate_plan(
    payload: PlannerRequest,
    planner: Annotated[PlanningAgent, Depends(_get_planner)],
) -> ResearchPlan:
    return await planner.create_plan(payload)


@agent_router.post("/questions/generate", response_model=FollowUpQuestionSet)
async def generate_questions(
    payload: QuestionsRequest,
    questions_agent: Annotated[QuestionsAgent, Depends(_get_questions_agent)],
) -> FollowUpQuestionSet:
    return await questions_agent.generate(payload)


@agent_router.post("/reviewer/review", response_model=ReviewResult)
async def review_evidence(
    payload: ReviewerRequest,
    reviewer: Annotated[EvidenceReviewer, Depends(_get_reviewer)],
) -> ReviewResult:
    return await reviewer.review(payload)


@agent_router.post("/report/generate", response_model=ReportArtifact)
async def generate_report(
    payload: ReportRequest,
    report_agent: Annotated[ReportGenerationAgent, Depends(_get_report_agent)],
) -> ReportArtifact:
    return await report_agent.generate(payload)


@agent_router.post("/researcher/research", response_model=ResearchResult)
async def conduct_research(
    payload: ResearchRequest,
    researcher: Annotated[ResearchAgent, Depends(_get_research_agent)],
) -> ResearchResult:
    return await researcher.research(payload)


@router.post("/runs", response_model=ResearchRun, status_code=201)
async def create_run(
    payload: CreateRunRequest,
    idempotency_key: IdempotencyKey,
    principal: Annotated[Principal, Depends(_get_principal)],
    service: Annotated[RunControlService, Depends(_get_run_control)],
) -> ResearchRun:
    return await service.create_run(principal, payload, idempotency_key=idempotency_key)


@router.get("/runs/{run_id}", response_model=ResearchRun)
async def get_run(
    run_id: str,
    principal: Annotated[Principal, Depends(_get_principal)],
    service: Annotated[RunControlService, Depends(_get_run_control)],
) -> ResearchRun:
    return await service.get_run(principal, run_id)


@router.post("/runs/{run_id}/start", response_model=ResearchRun)
async def start_run(
    run_id: str,
    idempotency_key: IdempotencyKey,
    principal: Annotated[Principal, Depends(_get_principal)],
    service: Annotated[RunControlService, Depends(_get_run_control)],
    dispatcher: Annotated[JobDispatcher, Depends(_get_job_dispatcher)],
) -> ResearchRun:
    run = await service.start_run(principal, run_id, idempotency_key=idempotency_key)
    await dispatcher.dispatch(make_phase_job(run, JobPhase.CLARIFY))
    return run


@router.post("/runs/{run_id}/clarifications", response_model=ResearchRun)
async def update_clarification(
    run_id: str,
    payload: ClarificationAnswerRequest,
    idempotency_key: IdempotencyKey,
    principal: Annotated[Principal, Depends(_get_principal)],
    service: Annotated[RunControlService, Depends(_get_run_control)],
    dispatcher: Annotated[JobDispatcher, Depends(_get_job_dispatcher)],
) -> ResearchRun:
    run = await service.submit_clarification_answers(
        principal,
        run_id,
        payload.answers,
        round_number=payload.round_number,
        idempotency_key=idempotency_key,
    )
    await dispatcher.dispatch(make_phase_job(run, JobPhase.CLARIFY))
    return run


@router.put("/runs/{run_id}/plan", response_model=ResearchRun)
async def save_run_plan(
    run_id: str,
    payload: ResearchPlan,
    idempotency_key: IdempotencyKey,
    principal: Annotated[Principal, Depends(_get_principal)],
    service: Annotated[RunControlService, Depends(_get_run_control)],
) -> ResearchRun:
    return await service.save_plan(principal, run_id, payload, idempotency_key=idempotency_key)


@router.post("/runs/{run_id}/plan/approve", response_model=ResearchRun)
async def approve_run_plan(
    run_id: str,
    payload: PlanApprovalRequest,
    idempotency_key: IdempotencyKey,
    principal: Annotated[Principal, Depends(_get_principal)],
    service: Annotated[RunControlService, Depends(_get_run_control)],
    dispatcher: Annotated[JobDispatcher, Depends(_get_job_dispatcher)],
) -> ResearchRun:
    run = await service.approve_plan(principal, run_id, payload, idempotency_key=idempotency_key)
    await dispatcher.dispatch(make_phase_job(run, JobPhase.RESEARCH))
    return run


@router.post("/runs/{run_id}/cancel", response_model=ResearchRun)
async def cancel_run(
    run_id: str,
    idempotency_key: IdempotencyKey,
    principal: Annotated[Principal, Depends(_get_principal)],
    service: Annotated[RunControlService, Depends(_get_run_control)],
) -> ResearchRun:
    return await service.cancel(principal, run_id, idempotency_key=idempotency_key)


@router.get("/runs/{run_id}/events", response_model=RunEventPage)
async def get_run_events(
    run_id: str,
    principal: Annotated[Principal, Depends(_get_principal)],
    service: Annotated[RunControlService, Depends(_get_run_control)],
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> RunEventPage:
    return await service.get_events(principal, run_id, after_cursor=after, limit=limit)


@router.get(
    "/runs/{run_id}/report",
    response_class=PlainTextResponse,
    responses={409: {"description": "Report not ready"}},
)
async def get_run_report(
    run_id: str,
    principal: Annotated[Principal, Depends(_get_principal)],
    service: Annotated[RunControlService, Depends(_get_run_control)],
) -> PlainTextResponse:
    markdown = await service.get_report(principal, run_id)
    return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8")
