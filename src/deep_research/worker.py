import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from deep_research.agents.clarifier import ClarifierAgent
from deep_research.agents.planner import PlanningAgent
from deep_research.agents.questions import QuestionsAgent
from deep_research.agents.report import ReportGenerationAgent
from deep_research.agents.researcher import ResearchAgent, merge_research_results
from deep_research.agents.reviewer import (
    EvidenceReviewer,
    EvidenceReviewValidationError,
    compact_limitations,
)
from deep_research.contracts.clarification import ClarifierRequest
from deep_research.contracts.evidence import (
    BudgetUsage,
    EvidencePackage,
    ReviewerRequest,
    ReviewState,
)
from deep_research.contracts.jobs import JobDelivery, JobPhase, PhaseJob
from deep_research.contracts.orchestration import GraphCheckpoint, GraphNode
from deep_research.contracts.planning import DepthPreset, PlannerRequest, ResearchPlan
from deep_research.contracts.questions import (
    QuestionsRequest,
    ReportContext,
    ReportSectionSummary,
)
from deep_research.contracts.reporting import ReportRequest
from deep_research.contracts.research import (
    ResearchRequest,
    ResearchResult,
    ResearchTask,
    ResearchTaskStatus,
)
from deep_research.contracts.runs import FailureUpdate, Principal, ResearchRun, RunState
from deep_research.jobs.base import JobDispatcher
from deep_research.models.gateway import ModelInvocationError
from deep_research.orchestration.graph import BoundedResearchGraph
from deep_research.persistence.runs import ConcurrencyConflictError
from deep_research.services.runs import (
    IdempotencyConflictError,
    InvalidRunTransitionError,
    RunControlService,
)
from deep_research.tools.research import ResearchServiceConfigurationError


class CooperativeCancellation(RuntimeError):
    pass


class StaleJobError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerAgents:
    clarifier: ClarifierAgent
    planner: PlanningAgent
    researcher_factory: Callable[[Principal, str], ResearchAgent]
    reviewer: EvidenceReviewer
    report: ReportGenerationAgent
    questions: QuestionsAgent


class DurableWorker:
    """Consumes one graph phase at a time and persists only through RunControlService."""

    def __init__(
        self,
        control: RunControlService,
        dispatcher: JobDispatcher,
        agents: WorkerAgents,
        *,
        max_delivery_attempts: int = 5,
        graph: BoundedResearchGraph | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._control = control
        self._dispatcher = dispatcher
        self._agents = agents
        self._max_delivery_attempts = max_delivery_attempts
        self._graph = graph or BoundedResearchGraph()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic

    async def run_once(self, *, wait_seconds: int = 20) -> bool:
        delivery = await self._dispatcher.receive(wait_seconds=wait_seconds)
        if delivery is None:
            return False
        try:
            await self.handle(delivery.job)
        except (
            CooperativeCancellation,
            StaleJobError,
            InvalidRunTransitionError,
            IdempotencyConflictError,
        ):
            await self._dispatcher.acknowledge(delivery)
        except ConcurrencyConflictError:
            if delivery.delivery_count >= self._max_delivery_attempts:
                await self._fail_delivery(delivery, "worker_conflict_ceiling")
                await self._dispatcher.acknowledge(delivery)
            else:
                await self._dispatcher.retry(delivery)
        except ResearchServiceConfigurationError as exc:
            await self._fail_delivery(delivery, "research_service_configuration", exc)
            await self._dispatcher.acknowledge(delivery)
        except Exception as exc:
            if delivery.delivery_count >= self._max_delivery_attempts:
                await self._fail_delivery(delivery, "worker_retry_ceiling", exc)
                await self._dispatcher.acknowledge(delivery)
            else:
                await self._dispatcher.retry(delivery)
        else:
            await self._dispatcher.acknowledge(delivery)
        return True

    async def run_forever(self, *, stop: asyncio.Event | None = None) -> None:
        """Poll until cooperative shutdown; each delivery still reaches ack or retry."""
        while stop is None or not stop.is_set():
            await self.run_once(wait_seconds=20)

    async def handle(self, job: PhaseJob) -> None:
        principal = Principal(subject=job.owner_id, tenant_id=job.tenant_id)
        run = await self._control.get_run(principal, job.run_id)
        self._validate_job_binding(job, run)
        if run.state.terminal:
            return
        handlers = {
            JobPhase.CLARIFY: self._clarify,
            JobPhase.PLAN: self._plan,
            JobPhase.RESEARCH: self._research,
            JobPhase.REVIEW: self._review,
            JobPhase.REPORT: self._report,
            JobPhase.QUESTIONS: self._questions,
            JobPhase.FINALIZE: self._finalize,
        }
        await handlers[job.phase](principal, job, run)

    async def _clarify(self, principal: Principal, job: PhaseJob, run: ResearchRun) -> None:
        if run.state is not RunState.CLARIFYING:
            return
        await self._guard_active(principal, run.run_id)
        checkpoint = run.graph_checkpoint
        started = self._monotonic()
        decision = await self._invoke_with_cancellation(
            self._agents.clarifier.evaluate,
            ClarifierRequest(
                topic=run.topic,
                answers=checkpoint.submitted_clarification_answers,
                current_brief=checkpoint.current_proposed_brief or run.brief,
                round_number=min(3, checkpoint.clarification_round + 1),
                depth=run.depth.value,
            ),
            principal,
            run.run_id,
        )
        await self._guard_active(principal, run.run_id)
        checkpoint = checkpoint.model_copy(
            update={
                "clarification": decision,
                "clarification_round": min(3, checkpoint.clarification_round + 1),
                "pending_clarification_questions": decision.questions,
                "submitted_clarification_answers": [],
                "current_proposed_brief": decision.brief,
                "completed_nodes": (
                    _append_node(checkpoint, GraphNode.CLARIFIER)
                    if decision.status == "scope_ready"
                    else checkpoint.completed_nodes
                ),
            }
        )
        updated = await self._control.record_clarification(
            principal,
            run.run_id,
            decision.brief,
            scope_ready=decision.status == "scope_ready",
            idempotency_key=job.idempotency_key("clarifier"),
            checkpoint=checkpoint,
            budget_usage=_increment_budget(
                run.budget_usage,
                model_calls=1,
                elapsed_seconds=self._monotonic() - started,
            ),
            expected_revision=run.revision,
        )
        if decision.status == "scope_ready":
            await self._dispatch(updated, JobPhase.PLAN)

    async def _plan(self, principal: Principal, job: PhaseJob, run: ResearchRun) -> None:
        if run.state is not RunState.PLANNING or run.brief is None:
            return
        await self._guard_active(principal, run.run_id)
        started = self._monotonic()
        plan = await self._invoke_with_cancellation(
            self._agents.planner.create_plan,
            PlannerRequest(brief=run.brief, depth=run.depth, previous_plan=run.plan),
            principal,
            run.run_id,
        )
        await self._guard_active(principal, run.run_id)
        checkpoint = run.graph_checkpoint.model_copy(
            update={
                "completed_nodes": _append_node(run.graph_checkpoint, GraphNode.PLANNER),
                "plan_version": plan.version,
                "plan_hash": plan.content_hash,
                "research_results": {},
                "evidence": None,
                "review": None,
                "repair_round": 0,
                "report": None,
                "questions": None,
            }
        )
        await self._control.save_plan(
            principal,
            run.run_id,
            plan,
            idempotency_key=job.idempotency_key("planner-and-approval-interrupt"),
            checkpoint=checkpoint,
            budget_usage=_increment_budget(
                run.budget_usage,
                model_calls=1,
                elapsed_seconds=self._monotonic() - started,
            ),
            expected_revision=run.revision,
        )

    async def _research(self, principal: Principal, job: PhaseJob, run: ResearchRun) -> None:
        if run.state is not RunState.RESEARCHING:
            return
        self._control.assert_exact_plan_approved(run)
        self._validate_job_plan(job, run)
        assert run.plan is not None
        self._graph.build_research_graph(run.plan)
        checkpoint = run.graph_checkpoint
        is_repair = (
            checkpoint.review is not None
            and checkpoint.review.review_state is ReviewState.REPAIR_REQUIRED
            and checkpoint.review.repair_round == checkpoint.repair_round
        )
        if checkpoint.evidence is not None and not is_repair:
            await self._dispatch(run, JobPhase.REVIEW)
            return
        pending: list[tuple[str, ResearchTask, object | None]] = []
        if is_repair:
            for repair in checkpoint.review.retry_tasks:
                for workstream in run.plan.workstreams:
                    scoped_questions = sorted(
                        set(repair.research_question_ids).intersection(
                            workstream.research_question_ids
                        )
                    )
                    if not scoped_questions:
                        continue
                    task = ResearchTask.for_workstream(
                        run.plan,
                        workstream.id,
                        attempt=min(3, checkpoint.repair_round + 2),
                    )
                    scoped_sections = sorted(
                        set(repair.section_ids).intersection(task.section_ids)
                    )
                    if not scoped_sections:
                        continue
                    key = (
                        f"repair:{checkpoint.repair_round + 1}:"
                        f"{repair.task_id}:{workstream.id}"
                    )
                    if key in checkpoint.research_results:
                        continue
                    scoped_repair = repair.model_copy(
                        update={
                            "research_question_ids": scoped_questions,
                            "section_ids": scoped_sections,
                        }
                    )
                    pending.append((key, task, scoped_repair))
        else:
            completed_workstreams = {
                key.removeprefix("workstream:")
                for key in checkpoint.research_results
                if key.startswith("workstream:")
            }
            count = len(run.plan.workstreams)
            initial_source_budget = _initial_source_budget(run.plan)
            base_sources, extra_sources = divmod(initial_source_budget, count)
            for index, item in enumerate(run.plan.workstreams):
                if item.id in completed_workstreams:
                    continue
                source_allocation = base_sources + (1 if index < extra_sources else 0)
                pending.append(
                    (
                        f"workstream:{item.id}",
                        ResearchTask.for_workstream(
                            run.plan, item.id, max_sources=source_allocation
                        ),
                        None,
                    )
                )

        if is_repair and pending:
            remaining_sources = max(
                0,
                run.plan.budget.max_accepted_sources - run.budget_usage.fetched_sources,
            )
            base_sources, extra_sources = divmod(remaining_sources, len(pending))
            pending = [
                (
                    key,
                    task.model_copy(
                        update={
                            "max_sources": base_sources + (1 if index < extra_sources else 0)
                        }
                    ),
                    repair,
                )
                for index, (key, task, repair) in enumerate(pending)
            ]

        # Dependencies are scheduled in waves; each wave is bounded by the approved concurrency.
        while pending:
            current = await self._guard_active(principal, run.run_id)
            checkpoint = current.graph_checkpoint
            completed = {
                key.removeprefix("workstream:")
                for key in checkpoint.research_results
                if key.startswith("workstream:")
            }
            ready = [
                item
                for item in pending
                if item[2] is not None
                or set(
                    next(
                        ws for ws in run.plan.workstreams if ws.id == item[1].workstream_id
                    ).dependencies
                ).issubset(completed)
            ]
            if not ready:
                raise RuntimeError("research workstream DAG made no progress")
            ready = ready[: (1 if is_repair else run.plan.budget.research_concurrency)]
            base_usage = current.budget_usage
            if _research_budget_exhausted(current):
                results = [_exhausted_result(task, base_usage) for _, task, _ in ready]
                wave_elapsed = 0.0
            else:
                wave_started = self._monotonic()
                results = await asyncio.gather(
                    *(
                        self._invoke_research(
                            self._agents.researcher_factory(principal, run.run_id),
                            ResearchRequest(
                                plan=run.plan,
                                task=task,
                                repair_task=repair,
                                budget_usage=base_usage,
                            ),
                            principal,
                            run.run_id,
                        )
                        for _, task, repair in ready
                    )
                )
                wave_elapsed = self._monotonic() - wave_started
            for result_index, ((key, task, _), result) in enumerate(
                zip(ready, results, strict=True)
            ):
                current = await self._guard_active(principal, run.run_id)
                if key in current.graph_checkpoint.research_results:
                    continue
                usage = _merge_budget_delta(current.budget_usage, base_usage, result.budget_usage)
                if result_index == 0:
                    usage = _increment_budget(usage, elapsed_seconds=wave_elapsed)
                bounded = _clamp_budget(usage, run.plan)
                results_by_key = dict(current.graph_checkpoint.research_results)
                results_by_key[key] = result
                next_checkpoint = current.graph_checkpoint.model_copy(
                    update={
                        "research_results": results_by_key,
                        "limitations": list(
                            dict.fromkeys(
                                [
                                    *current.graph_checkpoint.limitations,
                                    *result.limitations,
                                ]
                            )
                        ),
                    }
                )
                try:
                    await self._control.checkpoint_worker_progress(
                        principal,
                        run.run_id,
                        next_checkpoint,
                        expected_revision=current.revision,
                        idempotency_key=f"worker:{job.job_id}:{key}",
                        event_type="research.workstream.completed",
                        event_payload={
                            "workstream_id": task.workstream_id,
                            "task_id": result.task_id,
                            "repair_round": next_checkpoint.repair_round,
                            "search_count": max(
                                0,
                                result.budget_usage.searches - base_usage.searches,
                            ),
                            "source_count": len(result.evidence.sources),
                            "claim_count": len(result.evidence.claims),
                        },
                        budget_usage=bounded,
                    )
                except ConcurrencyConflictError:
                    latest = await self._control.get_run(principal, run.run_id)
                    if key not in latest.graph_checkpoint.research_results:
                        raise
            pending = [item for item in pending if item not in ready]

        current = await self._guard_active(principal, run.run_id)
        checkpoint = current.graph_checkpoint
        evidence = merge_research_results(list(checkpoint.research_results.values()))
        next_checkpoint = checkpoint.model_copy(
            update={
                "evidence": evidence,
                "completed_nodes": _append_node(checkpoint, GraphNode.RESEARCH),
                "repair_round": checkpoint.repair_round + (1 if is_repair else 0),
            }
        )
        updated = await self._control.checkpoint_worker_progress(
            principal,
            run.run_id,
            next_checkpoint,
            expected_revision=current.revision,
            idempotency_key=job.idempotency_key(f"research-round-{next_checkpoint.repair_round}"),
            event_type="graph.node.completed",
            event_payload={
                "node": GraphNode.RESEARCH.value,
                "repair_round": next_checkpoint.repair_round,
            },
        )
        await self._dispatch(updated, JobPhase.REVIEW)

    async def _review(self, principal: Principal, job: PhaseJob, run: ResearchRun) -> None:
        self._validate_job_plan(job, run)
        if run.state is RunState.RESEARCHING:
            prior_review = run.graph_checkpoint.review
            if (
                prior_review is not None
                and prior_review.review_state is ReviewState.REPAIR_REQUIRED
                and prior_review.repair_round == run.graph_checkpoint.repair_round
            ):
                return
            run = await self._control.advance(
                principal,
                run.run_id,
                RunState.REVIEWING,
                idempotency_key=job.idempotency_key("enter-review"),
                event_payload={"node": GraphNode.REVIEWER.value},
            )
        if run.state is not RunState.REVIEWING or run.plan is None:
            return
        if run.graph_checkpoint.evidence is None:
            raise RuntimeError("review requires checkpointed evidence")
        await self._guard_active(principal, run.run_id)
        started = self._monotonic()
        review_request = ReviewerRequest(
            plan=run.plan,
            evidence=run.graph_checkpoint.evidence,
            budget_usage=run.budget_usage,
            repair_round=run.graph_checkpoint.repair_round,
        )
        fallback_used = False
        try:
            review = await self._invoke_with_cancellation(
                self._agents.reviewer.review,
                review_request,
                principal,
                run.run_id,
            )
        except (ModelInvocationError, EvidenceReviewValidationError) as exc:
            await self._guard_active(principal, run.run_id)
            review = EvidenceReviewer.deterministic_fallback(review_request, exc)
            fallback_used = True
        current = await self._guard_active(principal, run.run_id)
        checkpoint = current.graph_checkpoint.model_copy(
            update={
                "review": review,
                "completed_nodes": _append_node(current.graph_checkpoint, GraphNode.REVIEWER),
                "limitations": list(
                    compact_limitations(
                        [*review.limitations, *current.graph_checkpoint.limitations],
                        max_items=(20 if run.plan.budget.preset is DepthPreset.QUICK else 60),
                    )
                ),
            }
        )
        if review.review_state is ReviewState.REPAIR_REQUIRED:
            target = RunState.RESEARCHING
        elif review.review_state is ReviewState.REJECTED:
            target = RunState.REVIEWING
        else:
            target = RunState.GENERATING_REPORT
        updated = await self._control.checkpoint_worker_progress(
            principal,
            run.run_id,
            checkpoint,
            expected_revision=current.revision,
            idempotency_key=job.idempotency_key(f"review-{checkpoint.repair_round}"),
            event_type="graph.node.completed",
            event_payload={
                "node": GraphNode.REVIEWER.value,
                "review_state": review.review_state.value,
                "fallback_used": fallback_used,
            },
            target=target,
            budget_usage=_increment_budget(
                current.budget_usage,
                model_calls=1,
                elapsed_seconds=self._monotonic() - started,
            ),
        )
        if review.review_state is ReviewState.REJECTED:
            await self._control.fail(
                principal,
                run.run_id,
                FailureUpdate(
                    code="insufficient_evidence",
                    message=(
                        "Evidence could not pass deterministic review; report generation was "
                        "skipped. " + review.limitations[0]
                    ),
                ),
                idempotency_key=job.idempotency_key(
                    f"insufficient-evidence-{checkpoint.repair_round}"
                ),
            )
            return
        await self._dispatch(
            updated,
            JobPhase.RESEARCH if target is RunState.RESEARCHING else JobPhase.REPORT,
        )

    async def _report(self, principal: Principal, job: PhaseJob, run: ResearchRun) -> None:
        if run.state is not RunState.GENERATING_REPORT or run.plan is None:
            return
        self._validate_job_plan(job, run)
        checkpoint = run.graph_checkpoint
        if checkpoint.evidence is None or checkpoint.review is None:
            raise RuntimeError("report requires checkpointed evidence and review")
        started = self._monotonic()
        artifact = await self._invoke_with_cancellation(
            self._agents.report.generate,
            ReportRequest(
                plan=run.plan,
                evidence=checkpoint.evidence,
                review=checkpoint.review,
                budget_usage=run.budget_usage,
            ),
            principal,
            run.run_id,
        )
        current = await self._guard_active(principal, run.run_id)
        checkpoint = current.graph_checkpoint.model_copy(
            update={
                "report": artifact,
                "completed_nodes": _append_node(current.graph_checkpoint, GraphNode.REPORT),
            }
        )
        updated = await self._control.checkpoint_worker_progress(
            principal,
            run.run_id,
            checkpoint,
            expected_revision=current.revision,
            idempotency_key=job.idempotency_key("report"),
            event_type="graph.node.completed",
            event_payload={"node": GraphNode.REPORT.value, "checksum": artifact.checksum},
            target=RunState.GENERATING_QUESTIONS,
            budget_usage=_increment_budget(
                current.budget_usage,
                model_calls=1,
                elapsed_seconds=self._monotonic() - started,
            ),
        )
        await self._dispatch(updated, JobPhase.QUESTIONS)

    async def _questions(self, principal: Principal, job: PhaseJob, run: ResearchRun) -> None:
        if run.state is not RunState.GENERATING_QUESTIONS or run.plan is None:
            return
        self._validate_job_plan(job, run)
        artifact = run.graph_checkpoint.report
        if artifact is None:
            raise RuntimeError("questions require a checkpointed report")
        started = self._monotonic()
        questions = await self._invoke_with_cancellation(
            self._agents.questions.generate,
            QuestionsRequest(report=_report_context(artifact.markdown, run), count=7),
            principal,
            run.run_id,
        )
        current = await self._guard_active(principal, run.run_id)
        checkpoint = current.graph_checkpoint.model_copy(
            update={
                "questions": questions,
                "completed_nodes": _append_node(current.graph_checkpoint, GraphNode.QUESTIONS),
            }
        )
        updated = await self._control.checkpoint_worker_progress(
            principal,
            run.run_id,
            checkpoint,
            expected_revision=current.revision,
            idempotency_key=job.idempotency_key("questions"),
            event_type="graph.node.completed",
            event_payload={"node": GraphNode.QUESTIONS.value, "count": len(questions.questions)},
            budget_usage=_increment_budget(
                current.budget_usage,
                model_calls=1,
                elapsed_seconds=self._monotonic() - started,
            ),
        )
        await self._dispatch(updated, JobPhase.FINALIZE)

    async def _finalize(self, principal: Principal, job: PhaseJob, run: ResearchRun) -> None:
        if run.state is not RunState.GENERATING_QUESTIONS or run.plan is None:
            return
        self._validate_job_plan(job, run)
        checkpoint = run.graph_checkpoint
        if not all(
            (checkpoint.evidence, checkpoint.review, checkpoint.report, checkpoint.questions)
        ):
            raise RuntimeError("finalization requires every prior artifact")
        assert checkpoint.evidence is not None and checkpoint.review is not None
        assert checkpoint.report is not None
        if checkpoint.review.evidence_checksum != checkpoint.evidence.calculate_checksum():
            raise RuntimeError("review no longer matches checkpointed evidence")
        if checkpoint.report.generation_metadata.plan_hash != run.plan.content_hash:
            raise RuntimeError("report no longer matches the approved plan")
        if any(not item.valid for item in checkpoint.report.mermaid_validation):
            raise RuntimeError("invalid Mermaid artifact cannot be finalized")
        source_ids = {source.source_id for source in checkpoint.evidence.sources}
        if not set(checkpoint.report.cited_source_ids).issubset(source_ids):
            raise RuntimeError("report contains unresolved citations")
        checkpoint = checkpoint.model_copy(
            update={"completed_nodes": _append_node(checkpoint, GraphNode.FINALIZE)}
        )
        await self._control.checkpoint_worker_progress(
            principal,
            run.run_id,
            checkpoint,
            expected_revision=run.revision,
            idempotency_key=job.idempotency_key("finalize"),
            event_type="run.completed",
            event_payload={
                "state": RunState.COMPLETED.value,
                "report_checksum": checkpoint.report.checksum,
            },
            target=RunState.COMPLETED,
        )

    async def _invoke_research(
        self,
        agent: ResearchAgent,
        request: ResearchRequest,
        principal: Principal,
        run_id: str,
    ) -> ResearchResult:
        method = agent.research
        if "cancellation_check" in inspect.signature(method).parameters:
            return await method(
                request,
                cancellation_check=lambda: self._guard_active(principal, run_id),
            )
        await self._guard_active(principal, run_id)
        result = await method(request)
        await self._guard_active(principal, run_id)
        return result

    async def _invoke_with_cancellation(
        self,
        method: Callable[..., Awaitable[Any]],
        request: object,
        principal: Principal,
        run_id: str,
    ) -> Any:
        if "cancellation_check" in inspect.signature(method).parameters:
            return await method(
                request,
                cancellation_check=lambda: self._guard_active(principal, run_id),
            )
        await self._guard_active(principal, run_id)
        result = await method(request)
        await self._guard_active(principal, run_id)
        return result

    async def _guard_active(self, principal: Principal, run_id: str) -> ResearchRun:
        run = await self._control.get_run(principal, run_id)
        if run.state is RunState.CANCELLED:
            raise CooperativeCancellation("run was cancelled")
        if run.state.terminal:
            raise StaleJobError(f"run is terminal: {run.state.value}")
        return run

    @staticmethod
    def _validate_job_binding(job: PhaseJob, run: ResearchRun) -> None:
        if job.owner_id != run.owner_id or job.tenant_id != run.tenant_id:
            raise StaleJobError("job ownership does not match the run")
        if job.expected_revision > run.revision:
            raise StaleJobError("job expects a future run revision")

    @staticmethod
    def _validate_job_plan(job: PhaseJob, run: ResearchRun) -> None:
        if (
            run.plan is None
            or job.plan_version != run.plan.version
            or job.plan_hash != run.plan.content_hash
            or run.approved_plan_version != job.plan_version
            or run.approved_plan_hash != job.plan_hash
        ):
            raise StaleJobError("job is not bound to the exact approved current plan")

    async def _dispatch(self, run: ResearchRun, phase: JobPhase) -> None:
        plan_bound = phase in {
            JobPhase.RESEARCH,
            JobPhase.REVIEW,
            JobPhase.REPORT,
            JobPhase.QUESTIONS,
            JobPhase.FINALIZE,
        }
        discriminator = f"{run.run_id}:{phase}:{run.revision}:{run.graph_checkpoint.repair_round}"
        await self._dispatcher.dispatch(
            PhaseJob(
                job_id=uuid5(NAMESPACE_URL, discriminator),
                phase=phase,
                run_id=run.run_id,
                tenant_id=run.tenant_id,
                owner_id=run.owner_id,
                expected_revision=run.revision,
                plan_version=run.plan.version if plan_bound and run.plan else None,
                plan_hash=run.plan.content_hash if plan_bound and run.plan else None,
                enqueued_at=self._clock(),
            )
        )

    async def _fail_delivery(
        self, delivery: JobDelivery, code: str, exc: Exception | None = None
    ) -> None:
        job = delivery.job
        principal = Principal(subject=job.owner_id, tenant_id=job.tenant_id)
        try:
            run = await self._control.get_run(principal, job.run_id)
            if not run.state.terminal:
                await self._control.fail(
                    principal,
                    job.run_id,
                    FailureUpdate(
                        code=code,
                        message=self._failure_message(delivery, code, exc),
                    ),
                    idempotency_key=job.idempotency_key(f"failure-{delivery.delivery_count}"),
                )
        except Exception:
            return

    def _failure_message(
        self, delivery: JobDelivery, code: str, exc: Exception | None
    ) -> str:
        context = (
            f"phase={delivery.job.phase.value}; "
            f"delivery_attempt={delivery.delivery_count}/{self._max_delivery_attempts}"
        )
        if exc is None:
            detail = code
        else:
            exception_detail = " ".join(str(exc).split())
            detail = type(exc).__name__
            if exception_detail:
                detail = f"{detail}: {exception_detail}"
        return f"{context}; {detail}"[:2_000]


def make_phase_job(run: ResearchRun, phase: JobPhase, *, now: datetime | None = None) -> PhaseJob:
    plan_bound = phase not in {JobPhase.CLARIFY, JobPhase.PLAN}
    discriminator = f"{run.run_id}:{phase}:{run.revision}:{run.graph_checkpoint.repair_round}"
    return PhaseJob(
        job_id=uuid5(NAMESPACE_URL, discriminator),
        phase=phase,
        run_id=run.run_id,
        tenant_id=run.tenant_id,
        owner_id=run.owner_id,
        expected_revision=run.revision,
        plan_version=run.plan.version if plan_bound and run.plan else None,
        plan_hash=run.plan.content_hash if plan_bound and run.plan else None,
        enqueued_at=now or datetime.now(UTC),
    )


def _append_node(checkpoint: GraphCheckpoint, node: GraphNode) -> list[GraphNode]:
    return [*checkpoint.completed_nodes, node]


def _increment_budget(usage: BudgetUsage, **increments: int | float) -> BudgetUsage:
    values = usage.model_dump()
    for key, value in increments.items():
        values[key] += value
    return BudgetUsage.model_validate(values)


def _merge_budget_delta(
    current: BudgetUsage, base: BudgetUsage, result: BudgetUsage
) -> BudgetUsage:
    values = current.model_dump()
    for field in type(current).model_fields:
        values[field] += max(0, getattr(result, field) - getattr(base, field))
    return BudgetUsage.model_validate(values)


def _clamp_budget(usage: BudgetUsage, plan) -> BudgetUsage:
    return usage.model_copy(
        update={
            "elapsed_seconds": min(usage.elapsed_seconds, plan.budget.target_duration_seconds),
            "searches": min(
                usage.searches, plan.budget.absolute_search_query_ceiling
            ),
            "fetched_sources": min(usage.fetched_sources, plan.budget.max_accepted_sources),
        }
    )


def _research_budget_exhausted(run: ResearchRun) -> bool:
    assert run.plan is not None
    usage = run.budget_usage
    budget = run.plan.budget
    return (
        usage.elapsed_seconds >= budget.target_duration_seconds
        or usage.searches >= budget.absolute_search_query_ceiling
        or usage.fetched_sources >= budget.max_accepted_sources
    )


def _initial_source_budget(plan: ResearchPlan) -> int:
    if (
        plan.budget.preset is not DepthPreset.QUICK
        or plan.budget.reviewer_retries == 0
    ):
        return plan.budget.max_accepted_sources
    repair_reserve = min(2, plan.budget.max_accepted_sources - 1)
    return plan.budget.max_accepted_sources - repair_reserve


def _exhausted_result(task: ResearchTask, usage: BudgetUsage) -> ResearchResult:
    return ResearchResult(
        task_id=task.task_id,
        status=ResearchTaskStatus.COMPLETED,
        evidence=EvidencePackage(),
        budget_usage=usage,
        limitations=["The approved research budget was exhausted before this task started."],
    )


def _report_context(markdown: str, run: ResearchRun) -> ReportContext:
    assert run.plan is not None
    lines = markdown.splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), run.topic)
    compact = " ".join(line.strip() for line in lines if line.strip() and not line.startswith("#"))
    summary = compact[:2_000] or "The generated report contains no prose summary."
    sections = [
        ReportSectionSummary(
            section_id=section.id,
            title=section.title,
            summary=f"See the generated report section: {section.title}.",
        )
        for section in run.plan.outline
    ]
    return ReportContext(
        title=title,
        executive_summary=summary,
        sections=sections,
        conclusion=summary[-2_000:],
        limitations=run.graph_checkpoint.limitations[:20],
    )
