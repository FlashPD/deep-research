import asyncio
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from deep_research.agents.clarifier import ClarifierAgent
from deep_research.contracts.clarification import (
    AnswerType,
    ClarificationAnswer,
    ClarifierRequest,
)
from deep_research.contracts.jobs import JobPhase
from deep_research.contracts.research import FetchPageRequest, WebSearchOperation
from deep_research.contracts.runs import (
    CreateRunRequest,
    PlanApprovalRequest,
    Principal,
    RunState,
)
from deep_research.dev_app import build_local_runtime
from deep_research.models.gateway import ModelGateway
from deep_research.services.runs import StalePlanApprovalError
from deep_research.settings import AppSettings
from deep_research.tools.playwright_fetcher import PlaywrightPageFetcher
from deep_research.tools.tavily import TavilySearchAdapter
from deep_research.worker import make_phase_job

pytestmark = pytest.mark.live


class LiveProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


def _settings(*, tavily: bool = False) -> AppSettings:
    settings = AppSettings()
    try:
        settings.validate_model_credentials()
    except ValueError as exc:
        pytest.skip(str(exc))
    if tavily and (
        settings.tavily_api_key is None
        or not settings.tavily_api_key.get_secret_value().strip()
    ):
        pytest.skip("TAVILY_API_KEY is required")
    return settings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role", ["clarifier", "planner", "researcher", "reviewer", "report", "questions"]
)
async def test_each_model_role_returns_one_structured_response(role: str) -> None:
    settings = _settings()
    gateway = ModelGateway(settings.validate_model_credentials())

    result = await gateway.generate_structured(
        role=role,
        prompt='Return the single JSON value {"status":"ok"}.',
        output_type=LiveProbe,
        system_prompt="Return only the requested structured output. Do not use tools.",
    )

    assert result.status == "ok"


def test_live_provider_selection_is_exact_and_credentialed() -> None:
    settings = _settings()
    models = settings.validate_model_credentials()

    for role in models.roles:
        route = models.route_for(role)
        assert route[0][1].provider is settings.model_provider


@pytest.mark.asyncio
async def test_tavily_discovery_then_playwright_fetch() -> None:
    settings = _settings(tavily=True)
    assert settings.tavily_api_key is not None
    search = TavilySearchAdapter(
        settings.tavily_api_key.get_secret_value(), base_url=settings.tavily_base_url
    )
    fetcher = PlaywrightPageFetcher(headless=settings.playwright_headless)

    results = await search.search_web(
        WebSearchOperation(
            query="OpenAI API official documentation",
            research_question_ids=["official_docs"],
            domains=["openai.com"],
            max_results=3,
        )
    )
    if not results:
        pytest.fail("Tavily returned no official OpenAI documentation result")
    page = await fetcher.fetch_page(FetchPageRequest(url=results[0].url))

    assert page.content
    assert page.canonical_url.startswith("https://")


@pytest.mark.asyncio
async def test_live_clarifier_interrupts_for_an_ambiguous_topic() -> None:
    settings = _settings()
    clarifier = ClarifierAgent(ModelGateway(settings.validate_model_credentials()))

    decision = await clarifier.evaluate(ClarifierRequest(topic="Research this for me"))

    assert decision.status == "needs_clarification"
    assert decision.questions


@pytest.mark.asyncio
async def test_complete_quick_public_web_workflow_and_exact_approval() -> None:
    runtime = build_local_runtime(_settings(tavily=True))
    principal = Principal(subject="live-user", tenant_id="live-tenant")
    run, interrupted = await _prepare_plan(runtime, principal, "Research grid-scale storage")
    assert interrupted
    assert run.plan is not None

    with pytest.raises(StalePlanApprovalError):
        await runtime.control.approve_plan(
            principal,
            run.run_id,
            PlanApprovalRequest(version=run.plan.version, content_hash="0" * 64),
            idempotency_key="live-stale-approval",
        )
    run = await runtime.control.approve_plan(
        principal,
        run.run_id,
        PlanApprovalRequest(version=run.plan.version, content_hash=run.plan.content_hash),
        idempotency_key="live-exact-approval",
    )
    await runtime.dispatcher.dispatch(make_phase_job(run, JobPhase.RESEARCH))
    run = await _drain_to_terminal(runtime, principal, run.run_id)

    assert run.state is RunState.COMPLETED
    checkpoint = run.graph_checkpoint
    assert checkpoint.report is not None
    assert checkpoint.questions is not None
    assert checkpoint.evidence is not None
    assert checkpoint.report.cited_source_ids
    sources = {source.source_id: source for source in checkpoint.evidence.sources}
    for source_id in checkpoint.report.cited_source_ids:
        assert source_id in sources
        assert sources[source_id].canonical_url is not None


@pytest.mark.asyncio
async def test_cancellation_during_live_research_is_cooperative() -> None:
    runtime = build_local_runtime(_settings(tavily=True))
    principal = Principal(subject="live-cancel-user", tenant_id="live-tenant")
    run, _interrupted = await _prepare_plan(
        runtime, principal, "Compare current grid-scale battery technologies"
    )
    assert run.plan is not None
    run = await runtime.control.approve_plan(
        principal,
        run.run_id,
        PlanApprovalRequest(version=run.plan.version, content_hash=run.plan.content_hash),
        idempotency_key="live-cancel-approval",
    )
    await runtime.dispatcher.dispatch(make_phase_job(run, JobPhase.RESEARCH))

    execution = asyncio.create_task(runtime.worker.run_once(wait_seconds=1))
    await asyncio.sleep(0.1)
    await runtime.control.cancel(
        principal, run.run_id, idempotency_key="live-cancel-during-research"
    )
    await execution

    cancelled = await runtime.control.get_run(principal, run.run_id)
    assert cancelled.state is RunState.CANCELLED


async def _prepare_plan(runtime, principal: Principal, topic: str):
    run = await runtime.control.create_run(
        principal,
        CreateRunRequest(topic=topic, depth="quick"),
        idempotency_key=f"live-create-{principal.subject}",
    )
    run = await runtime.control.start_run(
        principal, run.run_id, idempotency_key=f"live-start-{principal.subject}"
    )
    await runtime.dispatcher.dispatch(make_phase_job(run, JobPhase.CLARIFY))
    interrupted = False
    for _ in range(20):
        run = await runtime.control.get_run(principal, run.run_id)
        checkpoint = run.graph_checkpoint
        if run.state.terminal:
            pytest.fail(f"live run failed during plan preparation: {run.failure_code}")
        if run.state is RunState.AWAITING_PLAN_APPROVAL:
            return run, interrupted
        if checkpoint.pending_clarification_questions:
            interrupted = True
            answers = [
                ClarificationAnswer(question_id=question.id, value=_answer(question))
                for question in checkpoint.pending_clarification_questions
                if question.required
            ]
            run = await runtime.control.submit_clarification_answers(
                principal,
                run.run_id,
                answers,
                round_number=checkpoint.clarification_round,
                idempotency_key=(
                    f"live-answers-{principal.subject}-{checkpoint.clarification_round}"
                ),
            )
            await runtime.dispatcher.dispatch(make_phase_job(run, JobPhase.CLARIFY))
        elif runtime.dispatcher.pending:
            await runtime.worker.run_once(wait_seconds=1)
        else:
            await asyncio.sleep(0.1)
    pytest.fail("live run did not reach plan approval within the bounded setup loop")


async def _drain_to_terminal(runtime, principal: Principal, run_id: str):
    for _ in range(100):
        run = await runtime.control.get_run(principal, run_id)
        if run.state.terminal:
            return run
        if runtime.dispatcher.pending:
            await runtime.worker.run_once(wait_seconds=1)
        else:
            await asyncio.sleep(0.1)
    pytest.fail("live Quick run did not terminate within the bounded worker loop")


def _answer(question):
    if question.expected_answer_type is AnswerType.CONFIRMATION:
        return True
    if question.expected_answer_type is AnswerType.SINGLE_SELECT:
        return question.options[0]
    if question.expected_answer_type is AnswerType.MULTI_SELECT:
        return [question.options[0]]
    if question.expected_answer_type is AnswerType.DATE_RANGE:
        return "The five most recent complete calendar years"
    return "A concise decision-oriented report for technical and business leaders"
