import asyncio

import pytest

from deep_research.agents.clarifier import ClarifierAgent
from deep_research.agents.planner import PlanningAgent
from deep_research.agents.questions import QuestionsAgent
from deep_research.agents.report import ReportGenerationAgent
from deep_research.agents.reviewer import EvidenceReviewer, EvidenceReviewValidationError
from deep_research.contracts.clarification import ClarificationDecision, ResearchBrief
from deep_research.contracts.evidence import BudgetUsage, CoverageStatus, ReviewState
from deep_research.contracts.jobs import JobPhase
from deep_research.contracts.orchestration import GraphNode
from deep_research.contracts.planning import BudgetLimits, DepthPreset, ResearchPlan
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
from deep_research.worker import (
    DurableWorker,
    WorkerAgents,
    _initial_source_budget,
    make_phase_job,
)
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


def test_budget_clamp_never_goes_below_the_persisted_usage() -> None:
    from deep_research.worker import _clamp_budget

    plan = ResearchPlan.finalize(
        draft=make_plan_draft(),
        brief=ResearchBrief(topic="EV market"),
        budget=BudgetLimits.for_preset(DepthPreset.DEEP),
        version=1,
    )
    target = plan.budget.target_duration_seconds
    fresh = _clamp_budget(
        BudgetUsage(elapsed_seconds=target + 150, searches=3), plan, floor=BudgetUsage()
    )
    assert fresh.elapsed_seconds == target, "a first overshoot is clamped to the target"

    # The review phase already pushed the persisted value past the target; a later research
    # checkpoint must not try to lower it (the control plane rejects decreasing budgets).
    persisted = BudgetUsage(elapsed_seconds=target + 100, searches=40, fetched_sources=60)
    later = _clamp_budget(
        BudgetUsage(elapsed_seconds=target + 160, searches=41, fetched_sources=61),
        plan,
        floor=persisted,
    )
    assert later.elapsed_seconds == target + 100
    assert later.searches == 41
    assert later.fetched_sources == 61


def test_quick_initial_research_reserves_two_source_slots_for_repair() -> None:
    quick_plan = ResearchPlan.finalize(
        draft=make_plan_draft(),
        brief=ResearchBrief(topic="EV market"),
        budget=BudgetLimits.for_preset(DepthPreset.QUICK),
        version=1,
    )

    assert _initial_source_budget(quick_plan) == 8


class RaisingClarifier:
    async def evaluate(self, request):
        raise RuntimeError("provider unavailable")


class EmptyResearcher:
    async def research(self, request, *, cancellation_check=None):
        if cancellation_check is not None:
            await cancellation_check()
        return ResearchResult(
            task_id=request.task.task_id,
            status=ResearchTaskStatus.COMPLETED,
            evidence=make_evidence_package().model_copy(
                update={"sources": [], "excerpts": [], "claims": []}
            ),
            budget_usage=request.budget_usage.model_copy(
                update={
                    "searches": request.plan.budget.max_search_queries,
                    "model_calls": request.budget_usage.model_calls + 1,
                }
            ),
            limitations=["No source material was captured within the approved task bounds."],
        )


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
    assert (
        await control.get_report(principal, run.run_id)
        == completed.graph_checkpoint.report.markdown
    )
    assert completed.graph_checkpoint.completed_nodes[-1] is GraphNode.FINALIZE
    assert completed.budget_usage.searches == 1
    assert completed.budget_usage.model_calls == 7
    assert research_calls == ["market_analysis"]


@pytest.mark.asyncio
async def test_empty_research_fails_without_dispatching_report() -> None:
    principal = Principal(subject="empty-user", tenant_id="empty-tenant")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    decision = ClarificationDecision(
        status="scope_ready",
        brief=ResearchBrief(topic="EV market", objective="Measure the market"),
        interpretation_summary="The objective and scope are sufficiently precise.",
    )
    review_draft = make_review_draft(
        coverage=CoverageStatus.MISSING,
        recommends_approval=False,
    ).model_copy(update={"source_scores": []})
    report_gateway = FakeGateway(make_report_draft())
    agents = WorkerAgents(
        clarifier=ClarifierAgent(FakeGateway(decision)),
        planner=PlanningAgent(FakeGateway(make_plan_draft())),
        researcher_factory=lambda _principal, _run_id: EmptyResearcher(),
        reviewer=EvidenceReviewer(FakeGateway(review_draft)),
        report=ReportGenerationAgent(report_gateway),
        questions=QuestionsAgent(FakeGateway(make_question_set(count=7))),
    )
    worker = DurableWorker(control, dispatcher, agents)
    run = await _started(control, principal)
    await dispatcher.dispatch(make_phase_job(run, JobPhase.CLARIFY))
    await _drain(worker, dispatcher)
    run = await control.get_run(principal, run.run_id)
    assert run.plan is not None
    run = await control.approve_plan(
        principal,
        run.run_id,
        PlanApprovalRequest(version=run.plan.version, content_hash=run.plan.content_hash),
        idempotency_key="approve-empty-plan",
    )
    await dispatcher.dispatch(make_phase_job(run, JobPhase.RESEARCH))

    await _drain(worker, dispatcher)

    failed = await control.get_run(principal, run.run_id)
    assert failed.state is RunState.FAILED
    assert failed.failure_code == "insufficient_evidence"
    assert failed.graph_checkpoint.report is None
    assert report_gateway.calls == []


@pytest.mark.asyncio
async def test_reviewer_failure_uses_partial_report_path_for_complete_evidence() -> None:
    principal = Principal(subject="fallback-user", tenant_id="fallback-tenant")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    base_agents = _agents([])
    agents = WorkerAgents(
        clarifier=base_agents.clarifier,
        planner=base_agents.planner,
        researcher_factory=base_agents.researcher_factory,
        reviewer=EvidenceReviewer(
            FakeGateway(EvidenceReviewValidationError("invalid structured review"))
        ),
        report=base_agents.report,
        questions=base_agents.questions,
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
        idempotency_key="approve-fallback-plan",
    )
    await dispatcher.dispatch(make_phase_job(run, JobPhase.RESEARCH))

    assert await worker.run_once(wait_seconds=1)
    assert await worker.run_once(wait_seconds=1)

    reviewed = await control.get_run(principal, run.run_id)
    assert reviewed.state is RunState.GENERATING_REPORT
    assert reviewed.graph_checkpoint.review is not None
    assert (
        reviewed.graph_checkpoint.review.review_state
        is ReviewState.APPROVED_WITH_LIMITATIONS
    )
    assert any(
        "Semantic evidence review was unavailable" in item
        for item in reviewed.graph_checkpoint.review.limitations
    )
    events = await control.get_events(principal, run.run_id)
    reviewer_events = [
        event
        for event in events.events
        if event.payload.get("node") == GraphNode.REVIEWER.value
        and event.event_type == "graph.node.completed"
    ]
    assert reviewer_events[-1].payload["fallback_used"] is True


@pytest.mark.asyncio
async def test_independent_workstreams_start_in_parallel() -> None:
    principal = Principal(subject="parallel-user", tenant_id="parallel-tenant")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    release = asyncio.Event()
    both_started = asyncio.Event()
    started_workstreams: list[str] = []

    class ConcurrentResearcher(FakeResearcher):
        async def research(self, request, *, cancellation_check=None):
            started_workstreams.append(request.task.workstream_id)
            if len(started_workstreams) == 2:
                both_started.set()
            await release.wait()
            return await super().research(request, cancellation_check=cancellation_check)

    draft = make_plan_draft()
    second_question = draft.questions[0].model_copy(update={"id": "adoption_rate"})
    second_workstream = draft.workstreams[0].model_copy(
        update={
            "id": "adoption_analysis",
            "research_question_ids": ["adoption_rate"],
            "dependencies": [],
            "candidate_queries": ["official EV adoption statistics"],
        }
    )
    parallel_draft = draft.model_copy(
        update={
            "questions": [*draft.questions, second_question],
            "workstreams": [*draft.workstreams, second_workstream],
            "outline": [
                draft.outline[0].model_copy(
                    update={
                        "research_question_ids": ["market_size", "adoption_rate"]
                    }
                )
            ],
        }
    )
    decision = ClarificationDecision(
        status="scope_ready",
        brief=ResearchBrief(topic="EV market", objective="Measure the market"),
        interpretation_summary="The objective and scope are sufficiently precise.",
    )
    agents = _agents([])
    agents = WorkerAgents(
        clarifier=ClarifierAgent(FakeGateway(decision)),
        planner=PlanningAgent(FakeGateway(parallel_draft)),
        researcher_factory=lambda _principal, _run_id: ConcurrentResearcher([]),
        reviewer=agents.reviewer,
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
        idempotency_key="approve-parallel-plan",
    )

    research = asyncio.create_task(worker.handle(make_phase_job(run, JobPhase.RESEARCH)))
    await asyncio.wait_for(both_started.wait(), timeout=1)
    assert set(started_workstreams) == {"market_analysis", "adoption_analysis"}
    release.set()
    await research


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


class ValidationFailingResearcher:
    async def research(self, request, *, cancellation_check=None):
        raise ValueError("research synthesis remained invalid after one repair")


class OutputLimitPlanner:
    async def create_plan(self, request, *, cancellation_check=None):
        from deep_research.models.config import ModelProvider
        from deep_research.models.gateway import ModelAttemptFailure, ModelInvocationError

        raise ModelInvocationError(
            "planner",
            [
                ModelAttemptFailure(
                    target_name="anthropic_direct",
                    provider=ModelProvider.ANTHROPIC,
                    model_id="planner-model",
                    attempt=1,
                    max_attempts=2,
                    error_type="MaxTokensReachedException",
                    message="Model stopped generating due to maximum token limit.",
                )
            ],
        )


class BillingRejectedResearcher:
    async def research(self, request, *, cancellation_check=None):
        from deep_research.models.config import ModelProvider
        from deep_research.models.gateway import ModelAttemptFailure, ModelInvocationError

        raise ModelInvocationError(
            "researcher",
            [
                ModelAttemptFailure(
                    target_name="anthropic_direct",
                    provider=ModelProvider.ANTHROPIC,
                    model_id="research-model",
                    attempt=1,
                    max_attempts=2,
                    error_type="BadRequestError",
                    message=(
                        "Error code: 400 - invalid_request_error: Your credit balance is too "
                        "low to access the Anthropic API."
                    ),
                )
            ],
        )


@pytest.mark.asyncio
async def test_provider_rejection_fails_the_run_without_respending_research_budget() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    factory_calls: list[str] = []

    def factory(_principal, run_id: str):
        factory_calls.append(run_id)
        return BillingRejectedResearcher()

    agents = _agents([])
    agents = WorkerAgents(
        clarifier=agents.clarifier,
        planner=agents.planner,
        researcher_factory=factory,
        reviewer=agents.reviewer,
        report=agents.report,
        questions=agents.questions,
    )
    worker = DurableWorker(control, dispatcher, agents, max_delivery_attempts=5)
    run = await _approved_research_run(control, dispatcher, worker, principal)
    await dispatcher.dispatch(make_phase_job(run, JobPhase.RESEARCH))

    await worker.run_once(wait_seconds=1)

    failed = await control.get_run(principal, run.run_id)
    assert failed.state is RunState.FAILED
    assert failed.failure_code == "model_provider_rejected"
    assert "delivery_attempt=1/5" in failed.failure_message
    assert "credit balance is too low" in failed.failure_message
    assert dispatcher.pending == 0
    assert len(factory_calls) == 1, "the research phase (and its searches) ran exactly once"


@pytest.mark.asyncio
async def test_model_output_limit_fails_the_run_without_redelivery() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    agents = _agents([])
    agents = WorkerAgents(
        clarifier=agents.clarifier,
        planner=OutputLimitPlanner(),
        researcher_factory=agents.researcher_factory,
        reviewer=agents.reviewer,
        report=agents.report,
        questions=agents.questions,
    )
    worker = DurableWorker(control, dispatcher, agents, max_delivery_attempts=5)
    run = await _started(control, principal)
    await worker.handle(make_phase_job(run, JobPhase.CLARIFY))

    await worker.run_once(wait_seconds=1)

    failed = await control.get_run(principal, run.run_id)
    assert failed.state is RunState.FAILED
    assert failed.failure_code == "model_output_limit"
    assert "delivery_attempt=1/5" in failed.failure_message
    assert "MaxTokensReachedException" in failed.failure_message
    assert dispatcher.pending == 0


async def _approved_research_run(control, dispatcher, worker, principal):
    run = await _started(control, principal)
    await worker.handle(make_phase_job(run, JobPhase.CLARIFY))
    planning = await dispatcher.receive(wait_seconds=1)
    assert planning is not None
    await worker.handle(planning.job)
    run = await control.get_run(principal, run.run_id)
    assert run.plan is not None
    return await control.approve_plan(
        principal,
        run.run_id,
        PlanApprovalRequest(version=run.plan.version, content_hash=run.plan.content_hash),
        idempotency_key="approve-for-research",
    )


@pytest.mark.asyncio
async def test_validation_errors_fail_the_run_on_the_first_delivery() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    agents = _agents([])
    agents = WorkerAgents(
        clarifier=agents.clarifier,
        planner=agents.planner,
        researcher_factory=lambda _principal, _run_id: ValidationFailingResearcher(),
        reviewer=agents.reviewer,
        report=agents.report,
        questions=agents.questions,
    )
    worker = DurableWorker(control, dispatcher, agents, max_delivery_attempts=5)
    run = await _approved_research_run(control, dispatcher, worker, principal)
    await dispatcher.dispatch(make_phase_job(run, JobPhase.RESEARCH))

    await worker.run_once(wait_seconds=1)

    failed = await control.get_run(principal, run.run_id)
    assert failed.state is RunState.FAILED
    assert failed.failure_code == "worker_validation_error"
    assert failed.failure_message == (
        "phase=research; delivery_attempt=1/5; "
        "ValueError: research synthesis remained invalid after one repair"
    )
    assert dispatcher.pending == 0, "deterministic failures are not redelivered"


def _parallel_plan_draft():
    draft = make_plan_draft()
    second_question = draft.questions[0].model_copy(update={"id": "adoption_rate"})
    second_workstream = draft.workstreams[0].model_copy(
        update={
            "id": "adoption_analysis",
            "research_question_ids": ["adoption_rate"],
            "dependencies": [],
            "candidate_queries": ["official EV adoption statistics"],
        }
    )
    return draft.model_copy(
        update={
            "questions": [*draft.questions, second_question],
            "workstreams": [*draft.workstreams, second_workstream],
            "outline": [
                draft.outline[0].model_copy(
                    update={"research_question_ids": ["market_size", "adoption_rate"]}
                )
            ],
        }
    )


class OneWorkstreamFails(FakeResearcher):
    async def research(self, request, *, cancellation_check=None):
        if request.task.workstream_id == "adoption_analysis":
            raise RuntimeError("provider hiccup on the second workstream")
        return await super().research(request, cancellation_check=cancellation_check)


@pytest.mark.asyncio
async def test_completed_workstreams_are_checkpointed_before_a_wave_failure_surfaces() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    base = _agents([])
    agents = WorkerAgents(
        clarifier=base.clarifier,
        planner=PlanningAgent(FakeGateway(_parallel_plan_draft())),
        researcher_factory=lambda _principal, _run_id: OneWorkstreamFails([]),
        reviewer=base.reviewer,
        report=base.report,
        questions=base.questions,
    )
    worker = DurableWorker(control, dispatcher, agents)
    run = await _approved_research_run(control, dispatcher, worker, principal)

    with pytest.raises(RuntimeError, match="second workstream"):
        await worker.handle(make_phase_job(run, JobPhase.RESEARCH))

    latest = await control.get_run(principal, run.run_id)
    assert set(latest.graph_checkpoint.research_results) == {"workstream:market_analysis"}
    assert latest.state is RunState.RESEARCHING
    events = await control.get_events(principal, run.run_id)
    completed = [e for e in events.events if e.event_type == "research.workstream.completed"]
    assert [e.payload["workstream_id"] for e in completed] == ["market_analysis"]


@pytest.mark.asyncio
async def test_one_researcher_serves_every_parallel_workstream_in_a_phase() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    draft = make_plan_draft()
    second_question = draft.questions[0].model_copy(update={"id": "adoption_rate"})
    second_workstream = draft.workstreams[0].model_copy(
        update={
            "id": "adoption_analysis",
            "research_question_ids": ["adoption_rate"],
            "dependencies": [],
            "candidate_queries": ["official EV adoption statistics"],
        }
    )
    parallel_draft = draft.model_copy(
        update={
            "questions": [*draft.questions, second_question],
            "workstreams": [*draft.workstreams, second_workstream],
            "outline": [
                draft.outline[0].model_copy(
                    update={"research_question_ids": ["market_size", "adoption_rate"]}
                )
            ],
        }
    )
    factory_calls: list[str] = []
    research_calls: list[str] = []

    def counting_factory(_principal, run_id: str):
        factory_calls.append(run_id)
        return FakeResearcher(research_calls)

    base = _agents([])
    agents = WorkerAgents(
        clarifier=base.clarifier,
        planner=PlanningAgent(FakeGateway(parallel_draft)),
        researcher_factory=counting_factory,
        reviewer=base.reviewer,
        report=base.report,
        questions=base.questions,
    )
    worker = DurableWorker(control, dispatcher, agents)
    run = await _approved_research_run(control, dispatcher, worker, principal)

    await worker.handle(make_phase_job(run, JobPhase.RESEARCH))

    assert set(research_calls) == {"market_analysis", "adoption_analysis"}
    assert factory_calls == [run.run_id], "the run-bound adapter (and its page cache) is shared"


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
    assert failed.failure_message == (
        "phase=clarify; delivery_attempt=2/2; RuntimeError: provider unavailable"
    )


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
