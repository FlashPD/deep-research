import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from deep_research.contracts.clarification import ResearchBrief
from deep_research.contracts.evidence import BudgetUsage
from deep_research.contracts.orchestration import GraphCheckpoint, GraphNode
from deep_research.contracts.planning import ResearchPlan
from deep_research.contracts.runs import (
    CreateRunRequest,
    FailureUpdate,
    IdempotencyRecord,
    PlanApprovalRequest,
    Principal,
    ResearchRun,
    RunEvent,
    RunEventPage,
    RunState,
)
from deep_research.persistence.runs import (
    ConcurrencyConflictError,
    RunRepository,
)

_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.DRAFT: {RunState.CLARIFYING},
    RunState.CLARIFYING: {RunState.PLANNING},
    RunState.PLANNING: {RunState.AWAITING_PLAN_APPROVAL},
    RunState.AWAITING_PLAN_APPROVAL: {RunState.PLANNING, RunState.RESEARCHING},
    RunState.RESEARCHING: {RunState.REVIEWING},
    RunState.REVIEWING: {RunState.RESEARCHING, RunState.GENERATING_REPORT},
    RunState.GENERATING_REPORT: {RunState.GENERATING_QUESTIONS},
    RunState.GENERATING_QUESTIONS: {RunState.COMPLETED},
}
_TERMINAL_ALTERNATIVES = {RunState.FAILED, RunState.CANCELLED, RunState.EXPIRED}


class RunCommandConflictError(ValueError):
    pass


class InvalidRunTransitionError(RunCommandConflictError):
    pass


class StalePlanApprovalError(RunCommandConflictError):
    pass


class IdempotencyConflictError(RunCommandConflictError):
    pass


class RunControlService:
    def __init__(
        self,
        repository: RunRepository,
        *,
        retention_days: int = 30,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if retention_days < 1:
            raise ValueError("retention_days must be positive")
        self._repository = repository
        self._retention = timedelta(days=retention_days)
        self._clock = clock or (lambda: datetime.now(UTC))

    async def create_run(
        self,
        principal: Principal,
        request: CreateRunRequest,
        *,
        idempotency_key: str,
    ) -> ResearchRun:
        fingerprint = _fingerprint("create_run", request.model_dump(mode="json"))
        if replay := await self._replay(principal, idempotency_key, fingerprint):
            return replay
        now = self._now()
        run = ResearchRun(
            run_id=uuid.uuid4().hex,
            owner_id=principal.subject,
            tenant_id=principal.tenant_id,
            topic=request.topic,
            depth=request.depth,
            report_preferences=request.report_preferences,
            state=RunState.DRAFT,
            revision=1,
            next_event_cursor=2,
            created_at=now,
            updated_at=now,
            expires_at=now + self._retention,
        )
        event = RunEvent(
            run_id=run.run_id,
            cursor=1,
            event_type="run.created",
            timestamp=now,
            payload={"state": run.state.value, "depth": run.depth.value},
        )
        record = self._idempotency(principal, idempotency_key, fingerprint, run)
        try:
            await self._repository.create(run, event, record)
        except ConcurrencyConflictError:
            if replay := await self._replay(principal, idempotency_key, fingerprint):
                return replay
            raise
        return run

    async def start_run(
        self, principal: Principal, run_id: str, *, idempotency_key: str
    ) -> ResearchRun:
        return await self._change_state(
            principal,
            run_id,
            target=RunState.CLARIFYING,
            event_type="run.started",
            operation="start_run",
            idempotency_key=idempotency_key,
        )

    async def record_clarification(
        self,
        principal: Principal,
        run_id: str,
        brief: ResearchBrief,
        *,
        scope_ready: bool,
        idempotency_key: str,
        checkpoint: GraphCheckpoint | None = None,
        budget_usage: BudgetUsage | None = None,
        expected_revision: int | None = None,
    ) -> ResearchRun:
        payload = {
            "run_id": run_id,
            "brief": brief.model_dump(mode="json"),
            "scope_ready": scope_ready,
        }
        fingerprint = _fingerprint("record_clarification", payload)
        if replay := await self._replay(principal, idempotency_key, fingerprint):
            return replay
        current = await self._repository.get(principal, run_id)
        if expected_revision is not None and current.revision != expected_revision:
            raise ConcurrencyConflictError("run revision changed while clarifier was executing")
        if current.state is not RunState.CLARIFYING:
            raise InvalidRunTransitionError("clarification is only valid while CLARIFYING")
        target = RunState.PLANNING if scope_ready else RunState.CLARIFYING
        if budget_usage is not None:
            _validate_monotonic_budget(current.budget_usage, budget_usage)
        if checkpoint is not None:
            checkpoint = checkpoint.model_copy(update={"checkpointed_at": self._now()})
        updated, event = self._mutation(
            current,
            state=target,
            event_type="clarification.completed" if scope_ready else "clarification.requested",
            event_payload={"state": target.value, "scope_ready": scope_ready},
            extra={
                "brief": brief,
                **({"graph_checkpoint": checkpoint} if checkpoint is not None else {}),
                **({"budget_usage": budget_usage} if budget_usage is not None else {}),
            },
        )
        return await self._commit(principal, current, updated, event, idempotency_key, fingerprint)

    async def save_plan(
        self,
        principal: Principal,
        run_id: str,
        plan: ResearchPlan,
        *,
        idempotency_key: str,
        checkpoint: GraphCheckpoint | None = None,
        budget_usage: BudgetUsage | None = None,
        expected_revision: int | None = None,
    ) -> ResearchRun:
        fingerprint = _fingerprint(
            "save_plan", {"run_id": run_id, "plan": plan.model_dump(mode="json")}
        )
        if replay := await self._replay(principal, idempotency_key, fingerprint):
            return replay
        current = await self._repository.get(principal, run_id)
        if expected_revision is not None and current.revision != expected_revision:
            raise ConcurrencyConflictError("run revision changed while planner was executing")
        if current.state not in {RunState.PLANNING, RunState.AWAITING_PLAN_APPROVAL}:
            raise InvalidRunTransitionError("a plan cannot be saved in the current state")
        if current.brief is None or plan.brief != current.brief:
            raise RunCommandConflictError("plan brief does not match the run's clarified brief")
        expected_version = 1 if current.plan is None else current.plan.version + 1
        if plan.version != expected_version:
            raise RunCommandConflictError(f"next plan version must be {expected_version}")
        if budget_usage is not None:
            _validate_monotonic_budget(current.budget_usage, budget_usage)
        checkpoint = (checkpoint or current.graph_checkpoint).model_copy(
            update={
                "plan_version": plan.version,
                "plan_hash": plan.content_hash,
                "research_results": {},
                "evidence": None,
                "review": None,
                "repair_round": 0,
                "report": None,
                "questions": None,
                "checkpointed_at": self._now(),
            }
        )
        updated, event = self._mutation(
            current,
            state=RunState.AWAITING_PLAN_APPROVAL,
            event_type="plan.saved",
            event_payload={
                "state": RunState.AWAITING_PLAN_APPROVAL.value,
                "version": plan.version,
                "content_hash": plan.content_hash,
            },
            extra={
                "plan": plan,
                "approved_plan_version": None,
                "approved_plan_hash": None,
                "graph_checkpoint": checkpoint,
                **({"budget_usage": budget_usage} if budget_usage is not None else {}),
            },
        )
        return await self._commit(principal, current, updated, event, idempotency_key, fingerprint)

    async def approve_plan(
        self,
        principal: Principal,
        run_id: str,
        approval: PlanApprovalRequest,
        *,
        idempotency_key: str,
    ) -> ResearchRun:
        fingerprint = _fingerprint(
            "approve_plan",
            {"run_id": run_id, "approval": approval.model_dump(mode="json")},
        )
        if replay := await self._replay(principal, idempotency_key, fingerprint):
            return replay
        current = await self._repository.get(principal, run_id)
        if current.state is not RunState.AWAITING_PLAN_APPROVAL or current.plan is None:
            raise InvalidRunTransitionError("run is not awaiting plan approval")
        if (
            approval.version != current.plan.version
            or approval.content_hash != current.plan.content_hash
        ):
            raise StalePlanApprovalError("plan version or content hash is stale")
        completed_nodes = list(current.graph_checkpoint.completed_nodes)
        if GraphNode.PLAN_APPROVAL not in completed_nodes:
            completed_nodes.append(GraphNode.PLAN_APPROVAL)
        checkpoint = current.graph_checkpoint.model_copy(
            update={
                "completed_nodes": completed_nodes,
                "plan_version": current.plan.version,
                "plan_hash": current.plan.content_hash,
                "checkpointed_at": self._now(),
            }
        )
        updated, event = self._mutation(
            current,
            state=RunState.RESEARCHING,
            event_type="plan.approved",
            event_payload={
                "state": RunState.RESEARCHING.value,
                "version": approval.version,
                "content_hash": approval.content_hash,
            },
            extra={
                "approved_plan_version": approval.version,
                "approved_plan_hash": approval.content_hash,
                "graph_checkpoint": checkpoint,
            },
        )
        return await self._commit(principal, current, updated, event, idempotency_key, fingerprint)

    async def advance(
        self,
        principal: Principal,
        run_id: str,
        target: RunState,
        *,
        idempotency_key: str,
        event_payload: dict[str, Any] | None = None,
    ) -> ResearchRun:
        return await self._change_state(
            principal,
            run_id,
            target=target,
            event_type=f"run.{target.value.casefold()}",
            operation=f"advance_{target.value}",
            idempotency_key=idempotency_key,
            event_payload=event_payload,
        )

    async def record_budget(
        self,
        principal: Principal,
        run_id: str,
        budget_usage: BudgetUsage,
        *,
        idempotency_key: str,
    ) -> ResearchRun:
        fingerprint = _fingerprint(
            "record_budget",
            {"run_id": run_id, "budget_usage": budget_usage.model_dump(mode="json")},
        )
        if replay := await self._replay(principal, idempotency_key, fingerprint):
            return replay
        current = await self._repository.get(principal, run_id)
        if current.state.terminal:
            raise InvalidRunTransitionError("terminal run budgets cannot be changed")
        _validate_monotonic_budget(current.budget_usage, budget_usage)
        updated, event = self._mutation(
            current,
            state=current.state,
            event_type="budget.updated",
            event_payload=budget_usage.model_dump(mode="json"),
            extra={"budget_usage": budget_usage},
        )
        return await self._commit(principal, current, updated, event, idempotency_key, fingerprint)

    async def checkpoint_worker_progress(
        self,
        principal: Principal,
        run_id: str,
        checkpoint: GraphCheckpoint,
        *,
        expected_revision: int,
        idempotency_key: str,
        event_type: str,
        event_payload: dict[str, Any],
        target: RunState | None = None,
        budget_usage: BudgetUsage | None = None,
    ) -> ResearchRun:
        """Atomically persist one worker node result through the control plane.

        The explicit revision is captured before external work starts. It prevents a slow
        worker from overwriting cancellation, approval edits, or a newer worker checkpoint.
        """
        _validate_safe_event_payload(event_payload)
        fingerprint = _fingerprint(
            "checkpoint_worker_progress",
            {
                "run_id": run_id,
                "event_type": event_type,
                "event_payload": event_payload,
                "target": target.value if target else None,
            },
        )
        if replay := await self._replay(principal, idempotency_key, fingerprint):
            return replay
        current = await self._repository.get(principal, run_id)
        if current.revision != expected_revision:
            raise ConcurrencyConflictError("run revision changed while worker node was executing")
        next_state = target or current.state
        if target is not None and target is not current.state:
            _validate_transition(current.state, target)
        if budget_usage is not None:
            _validate_monotonic_budget(current.budget_usage, budget_usage)
        if next_state in {
            RunState.RESEARCHING,
            RunState.REVIEWING,
            RunState.GENERATING_REPORT,
            RunState.GENERATING_QUESTIONS,
            RunState.COMPLETED,
        }:
            self.assert_exact_plan_approved(current)
        checkpoint = checkpoint.model_copy(update={"checkpointed_at": self._now()})
        extra: dict[str, Any] = {"graph_checkpoint": checkpoint}
        if budget_usage is not None:
            extra["budget_usage"] = budget_usage
        updated, event = self._mutation(
            current,
            state=next_state,
            event_type=event_type,
            event_payload=event_payload,
            extra=extra,
        )
        return await self._commit(principal, current, updated, event, idempotency_key, fingerprint)

    @staticmethod
    def assert_exact_plan_approved(run: ResearchRun) -> None:
        if (
            run.plan is None
            or run.approved_plan_version != run.plan.version
            or run.approved_plan_hash != run.plan.content_hash
        ):
            raise StalePlanApprovalError(
                "public-web research requires approval of the exact current plan"
            )

    async def cancel(
        self, principal: Principal, run_id: str, *, idempotency_key: str
    ) -> ResearchRun:
        return await self._change_state(
            principal,
            run_id,
            target=RunState.CANCELLED,
            event_type="run.cancelled",
            operation="cancel_run",
            idempotency_key=idempotency_key,
        )

    async def fail(
        self,
        principal: Principal,
        run_id: str,
        failure: FailureUpdate,
        *,
        idempotency_key: str,
    ) -> ResearchRun:
        fingerprint = _fingerprint(
            "fail_run", {"run_id": run_id, "failure": failure.model_dump(mode="json")}
        )
        if replay := await self._replay(principal, idempotency_key, fingerprint):
            return replay
        current = await self._repository.get(principal, run_id)
        _validate_transition(current.state, RunState.FAILED)
        updated, event = self._mutation(
            current,
            state=RunState.FAILED,
            event_type="run.failed",
            event_payload={"state": RunState.FAILED.value, "code": failure.code},
            extra={
                "failure_code": failure.code,
                "failure_message": failure.message,
            },
        )
        return await self._commit(principal, current, updated, event, idempotency_key, fingerprint)

    async def get_run(self, principal: Principal, run_id: str) -> ResearchRun:
        return await self._repository.get(principal, run_id)

    async def get_events(
        self,
        principal: Principal,
        run_id: str,
        *,
        after_cursor: int = 0,
        limit: int = 100,
    ) -> RunEventPage:
        events = await self._repository.list_events(
            principal, run_id, after_cursor=after_cursor, limit=limit
        )
        next_cursor = events[-1].cursor if events else after_cursor
        return RunEventPage(events=events, next_cursor=next_cursor)

    async def _change_state(
        self,
        principal: Principal,
        run_id: str,
        *,
        target: RunState,
        event_type: str,
        operation: str,
        idempotency_key: str,
        event_payload: dict[str, Any] | None = None,
    ) -> ResearchRun:
        fingerprint = _fingerprint(
            operation,
            {
                "run_id": run_id,
                "target": target.value,
                "event_payload": event_payload or {},
            },
        )
        if replay := await self._replay(principal, idempotency_key, fingerprint):
            return replay
        current = await self._repository.get(principal, run_id)
        _validate_transition(current.state, target)
        updated, event = self._mutation(
            current,
            state=target,
            event_type=event_type,
            event_payload={"state": target.value, **(event_payload or {})},
        )
        return await self._commit(principal, current, updated, event, idempotency_key, fingerprint)

    async def _commit(
        self,
        principal: Principal,
        current: ResearchRun,
        updated: ResearchRun,
        event: RunEvent,
        idempotency_key: str,
        fingerprint: str,
    ) -> ResearchRun:
        record = self._idempotency(principal, idempotency_key, fingerprint, updated)
        try:
            await self._repository.commit(
                updated,
                event,
                expected_revision=current.revision,
                idempotency=record,
            )
        except ConcurrencyConflictError:
            if replay := await self._replay(principal, idempotency_key, fingerprint):
                return replay
            raise
        return updated

    def _mutation(
        self,
        current: ResearchRun,
        *,
        state: RunState,
        event_type: str,
        event_payload: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> tuple[ResearchRun, RunEvent]:
        now = self._now()
        values = {
            "state": state,
            "revision": current.revision + 1,
            "next_event_cursor": current.next_event_cursor + 1,
            "updated_at": now,
            **(extra or {}),
        }
        updated = ResearchRun.model_validate({**current.model_dump(mode="python"), **values})
        event = RunEvent(
            run_id=current.run_id,
            cursor=current.next_event_cursor,
            event_type=event_type,
            timestamp=now,
            payload=event_payload,
        )
        return updated, event

    async def _replay(self, principal: Principal, key: str, fingerprint: str) -> ResearchRun | None:
        record = await self._repository.get_idempotency(principal, key)
        if record is None:
            return None
        if record.fingerprint != fingerprint:
            raise IdempotencyConflictError(
                "idempotency key was already used for a different command"
            )
        return ResearchRun.model_validate_json(record.response_json)

    def _idempotency(
        self,
        principal: Principal,
        key: str,
        fingerprint: str,
        response: ResearchRun,
    ) -> IdempotencyRecord:
        return IdempotencyRecord(
            tenant_id=principal.tenant_id,
            owner_id=principal.subject,
            key=key,
            fingerprint=fingerprint,
            response_json=response.model_dump_json(),
            expires_at=response.expires_at,
        )

    def _now(self) -> datetime:
        value = self._clock()
        return value if value.tzinfo else value.replace(tzinfo=UTC)


def _validate_transition(current: RunState, target: RunState) -> None:
    if current.terminal:
        raise InvalidRunTransitionError(f"terminal run {current.value} cannot transition")
    if target in _TERMINAL_ALTERNATIVES:
        return
    if target not in _TRANSITIONS.get(current, set()):
        raise InvalidRunTransitionError(
            f"invalid run transition from {current.value} to {target.value}"
        )


def _validate_monotonic_budget(current: BudgetUsage, updated: BudgetUsage) -> None:
    for field in type(current).model_fields:
        if getattr(updated, field) < getattr(current, field):
            raise RunCommandConflictError(f"budget field {field!r} cannot decrease")


def _validate_safe_event_payload(payload: dict[str, Any]) -> None:
    forbidden = {"prompt", "content", "excerpt", "token", "secret", "authorization"}

    def validate_keys(value: Any) -> None:
        if isinstance(value, dict):
            if any(str(key).casefold() in forbidden for key in value):
                raise ValueError(
                    "worker progress events may contain only safe display metadata"
                )
            for nested in value.values():
                validate_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                validate_keys(nested)

    validate_keys(payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode()) > 8_192:
        raise ValueError("worker progress event payload exceeds 8 KiB")


def _fingerprint(operation: str, payload: Any) -> str:
    canonical = json.dumps(
        {"operation": operation, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
