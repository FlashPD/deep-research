import asyncio

import pytest

from deep_research.agents.clarifier import ClarifierAgent
from deep_research.agents.planner import PlanningAgent
from deep_research.agents.questions import QuestionsAgent
from deep_research.agents.report import ReportGenerationAgent
from deep_research.agents.reviewer import EvidenceReviewer
from deep_research.contracts.clarification import ClarificationDecision, ResearchBrief
from deep_research.contracts.evidence import BudgetUsage, CoverageStatus, ReviewState
from deep_research.contracts.jobs import JobPhase
from deep_research.contracts.orchestration import GraphNode
from deep_research.contracts.research import ResearchResult, ResearchTaskStatus
from deep_research.contracts.runs import (
    CreateRunRequest,
    PlanApprovalRequest,
    Principal,
    RunState,
)
from deep_research.jobs.memory import InMemoryJobDispatcher
from deep_research.persistence.memory import InMemoryRunRepository
from deep_research.persistence.runs import ConcurrencyConflictError
from deep_research.services.runs import RunControlService
from deep_research.worker import DurableWorker, WorkerAgents, make_phase_job
from tests.conftest import FakeGateway
from tests.factories import (
    make_evidence_package,
    make_plan_draft,
    make_question_set,
    make_report_draft,
    make_review_draft,
)


class FakeResearcher:
    def __init__(
        self,
        calls: list[str],
        gate: asyncio.Event | None = None,
        started: asyncio.Event | None = None,
    ) -> None:
        self.calls = calls
        self.gate = gate
        self.started = started

    async def research(self, request, *, cancellation_check=None):
        self.calls.append(request.task.workstream_id)
        if self.started is not None:
            self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if cancellation_check is not None:
            await cancellation_check()
        usage = request.budget_usage.model_copy(
            update={
                "searches": request.budget_usage.searches + 1,
                "fetched_sources": request.budget_usage.fetched_sources + 1,
                "model_calls": request.budget_usage.model_calls + 2,
            }
        )
        return ResearchResult(
            task_id=request.task.task_id,
            status=ResearchTaskStatus.COMPLETED,
            evidence=make_evidence_package(),
            budget_usage=usage,
        )


class RaisingClarifier:
    async def evaluate(self, request):
        raise RuntimeError("provider unavailable")


def _agents(
    research_calls: list[str],
    gate: asyncio.Event | None = None,
    started: asyncio.Event | None = None,
) -> WorkerAgents:
    decision = ClarificationDecision(
        status="scope_ready",
        brief=ResearchBrief(topic="EV market", objective="Measure the market"),
        interpretation_summary="The objective and scope are sufficiently precise.",
    )
    return WorkerAgents(
        clarifier=ClarifierAgent(FakeGateway(decision)),
        planner=PlanningAgent(FakeGateway(make_plan_draft())),
        researcher_factory=lambda _principal, _run_id: FakeResearcher(
            research_calls, gate, started
        ),
        reviewer=EvidenceReviewer(FakeGateway(make_review_draft())),
        report=ReportGenerationAgent(FakeGateway(make_report_draft())),
        questions=QuestionsAgent(FakeGateway(make_question_set(count=7))),
    )


async def _started(control, principal):
    run = await control.create_run(
        principal,
        CreateRunRequest(topic="EV market"),
        idempotency_key="create-run",
    )
    return await control.start_run(principal, run.run_id, idempotency_key="start-run")


async def _drain(worker: DurableWorker, dispatcher: InMemoryJobDispatcher) -> None:
    while dispatcher.pending:
        await worker.run_once(wait_seconds=1)


@pytest.mark.asyncio
async def test_complete_mocked_workflow_is_checkpointed_and_finalized() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    research_calls: list[str] = []
    worker = DurableWorker(control, dispatcher, _agents(research_calls))
    run = await _started(control, principal)
    await dispatcher.dispatch(make_phase_job(run, JobPhase.CLARIFY))

    await _drain(worker, dispatcher)
    run = await control.get_run(principal, run.run_id)
    assert run.state is RunState.AWAITING_PLAN_APPROVAL
    assert run.graph_checkpoint.checkpointed_at is not None
    assert run.plan is not None

    run = await control.approve_plan(
        principal,
        run.run_id,
        PlanApprovalRequest(version=run.plan.version, content_hash=run.plan.content_hash),
        idempotency_key="approve-plan",
    )
    await dispatcher.dispatch(make_phase_job(run, JobPhase.RESEARCH))
    await _drain(worker, dispatcher)

    completed = await control.get_run(principal, run.run_id)
    assert completed.state is RunState.COMPLETED
    assert completed.graph_checkpoint.report is not None
    assert completed.graph_checkpoint.questions is not None
    assert completed.graph_checkpoint.completed_nodes[-1] is GraphNode.FINALIZE
    assert completed.budget_usage.searches == 1
    assert completed.budget_usage.model_calls == 7
    assert research_calls == ["market_analysis"]


@pytest.mark.asyncio
async def test_duplicate_delivery_does_not_duplicate_node_events() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    worker = DurableWorker(control, dispatcher, _agents([]))
    run = await _started(control, principal)
    job = make_phase_job(run, JobPhase.CLARIFY)

    await worker.handle(job)
    await worker.handle(job)

    events = await control.get_events(principal, run.run_id)
    assert [item.event_type for item in events.events].count("clarification.completed") == 1


@pytest.mark.asyncio
async def test_new_worker_resumes_from_the_last_completed_node() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    research_calls: list[str] = []
    first_worker = DurableWorker(control, dispatcher, _agents(research_calls))
    run = await _started(control, principal)
    await first_worker.handle(make_phase_job(run, JobPhase.CLARIFY))
    plan_delivery = await dispatcher.receive(wait_seconds=1)
    assert plan_delivery is not None
    await first_worker.handle(plan_delivery.job)
    run = await control.get_run(principal, run.run_id)
    assert run.plan is not None
    run = await control.approve_plan(
        principal,
        run.run_id,
        PlanApprovalRequest(version=run.plan.version, content_hash=run.plan.content_hash),
        idempotency_key="approve-plan",
    )
    await first_worker.handle(make_phase_job(run, JobPhase.RESEARCH))
    checkpointed = await control.get_run(principal, run.run_id)
    assert checkpointed.graph_checkpoint.evidence is not None

    # Simulate process replacement after research committed but before review was consumed.
    replacement = DurableWorker(control, dispatcher, _agents(research_calls))
    await _drain(replacement, dispatcher)

    assert (await control.get_run(principal, run.run_id)).state is RunState.COMPLETED
    assert research_calls == ["market_analysis"]


@pytest.mark.asyncio
async def test_public_research_job_before_approval_is_a_noop() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    calls: list[str] = []
    worker = DurableWorker(control, dispatcher, _agents(calls))
    run = await _started(control, principal)
    await worker.handle(make_phase_job(run, JobPhase.CLARIFY))
    plan_job = await dispatcher.receive(wait_seconds=1)
    assert plan_job is not None
    await worker.handle(plan_job.job)
    run = await control.get_run(principal, run.run_id)

    await worker.handle(make_phase_job(run, JobPhase.RESEARCH))

    assert calls == []
    assert (await control.get_run(principal, run.run_id)).state is RunState.AWAITING_PLAN_APPROVAL


@pytest.mark.asyncio
async def test_cancellation_during_research_prevents_a_late_checkpoint() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    gate = asyncio.Event()
    started = asyncio.Event()
    calls: list[str] = []
    worker = DurableWorker(control, dispatcher, _agents(calls, gate, started))
    run = await _started(control, principal)
    await worker.handle(make_phase_job(run, JobPhase.CLARIFY))
    queued = await dispatcher.receive(wait_seconds=1)
    assert queued is not None
    await worker.handle(queued.job)
    run = await control.get_run(principal, run.run_id)
    assert run.plan is not None
    run = await control.approve_plan(
        principal,
        run.run_id,
        PlanApprovalRequest(version=run.plan.version, content_hash=run.plan.content_hash),
        idempotency_key="approve-plan",
    )
    task = asyncio.create_task(worker.handle(make_phase_job(run, JobPhase.RESEARCH)))
    await started.wait()
    await control.cancel(principal, run.run_id, idempotency_key="cancel-run")
    gate.set()

    with pytest.raises(RuntimeError, match="cancelled"):
        await task
    cancelled = await control.get_run(principal, run.run_id)
    assert cancelled.state is RunState.CANCELLED
    assert cancelled.graph_checkpoint.research_results == {}


@pytest.mark.asyncio
async def test_stale_clarifier_cannot_overwrite_a_newer_revision() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    gate = asyncio.Event()

    class SlowClarifier:
        async def evaluate(self, request):
            await gate.wait()
            return ClarificationDecision(
                status="scope_ready",
                brief=ResearchBrief(topic=request.topic),
                interpretation_summary="Ready.",
            )

    agents = _agents([])
    agents = WorkerAgents(
        clarifier=SlowClarifier(),
        planner=agents.planner,
        researcher_factory=agents.researcher_factory,
        reviewer=agents.reviewer,
        report=agents.report,
        questions=agents.questions,
    )
    worker = DurableWorker(control, dispatcher, agents)
    run = await _started(control, principal)
    task = asyncio.create_task(worker.handle(make_phase_job(run, JobPhase.CLARIFY)))
    await asyncio.sleep(0)
    await control.record_budget(
        principal,
        run.run_id,
        BudgetUsage(model_calls=1),
        idempotency_key="newer-budget",
    )
    gate.set()

    with pytest.raises(ConcurrencyConflictError):
        await task
    latest = await control.get_run(principal, run.run_id)
    assert latest.state is RunState.CLARIFYING
    assert latest.graph_checkpoint.clarification is None


@pytest.mark.asyncio
async def test_delivery_retry_ceiling_fails_the_run_deterministically() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    agents = _agents([])
    agents = WorkerAgents(
        clarifier=RaisingClarifier(),
        planner=agents.planner,
        researcher_factory=agents.researcher_factory,
        reviewer=agents.reviewer,
        report=agents.report,
        questions=agents.questions,
    )
    worker = DurableWorker(control, dispatcher, agents, max_delivery_attempts=2)
    run = await _started(control, principal)
    await dispatcher.dispatch(make_phase_job(run, JobPhase.CLARIFY))

    await worker.run_once(wait_seconds=1)
    await worker.run_once(wait_seconds=1)

    failed = await control.get_run(principal, run.run_id)
    assert failed.state is RunState.FAILED
    assert failed.failure_code == "worker_retry_ceiling"


@pytest.mark.asyncio
async def test_targeted_repair_rounds_stop_at_the_plan_ceiling() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    research_calls: list[str] = []
    agents = _agents(research_calls)
    gap_with_retry = make_review_draft(
        coverage=CoverageStatus.PARTIAL,
        recommends_approval=False,
        with_retry=True,
    )
    gap_at_ceiling = make_review_draft(
        coverage=CoverageStatus.PARTIAL,
        recommends_approval=False,
    )
    agents = WorkerAgents(
        clarifier=agents.clarifier,
        planner=agents.planner,
        researcher_factory=agents.researcher_factory,
        reviewer=EvidenceReviewer(
            FakeGateway(gap_with_retry, gap_with_retry, gap_at_ceiling)
        ),
        report=agents.report,
        questions=agents.questions,
    )
    worker = DurableWorker(control, dispatcher, agents)
    run = await _started(control, principal)
    await worker.handle(make_phase_job(run, JobPhase.CLARIFY))
    planning = await dispatcher.receive(wait_seconds=1)
    assert planning is not None
    await worker.handle(planning.job)
    run = await control.get_run(principal, run.run_id)
    assert run.plan is not None
    run = await control.approve_plan(
        principal,
        run.run_id,
        PlanApprovalRequest(version=run.plan.version, content_hash=run.plan.content_hash),
        idempotency_key="approve-plan",
    )
    await dispatcher.dispatch(make_phase_job(run, JobPhase.RESEARCH))

    for _ in range(10):
        current = await control.get_run(principal, run.run_id)
        if current.state is RunState.GENERATING_REPORT:
            break
        assert await worker.run_once(wait_seconds=1)

    reviewed = await control.get_run(principal, run.run_id)
    assert reviewed.state is RunState.GENERATING_REPORT
    assert reviewed.graph_checkpoint.repair_round == 2
    assert reviewed.graph_checkpoint.review is not None
    assert reviewed.graph_checkpoint.review.review_state is ReviewState.APPROVED_WITH_LIMITATIONS
    assert len(research_calls) == 3
