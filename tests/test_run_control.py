import asyncio
from datetime import UTC, datetime

import pytest

from deep_research.contracts.clarification import ResearchBrief
from deep_research.contracts.evidence import BudgetUsage
from deep_research.contracts.planning import (
    BudgetLimits,
    DepthPreset,
    ReportSection,
    ResearchPlan,
    ResearchPlanDraft,
    ResearchQuestion,
    SourceStrategy,
    Workstream,
)
from deep_research.contracts.runs import (
    CreateRunRequest,
    PlanApprovalRequest,
    Principal,
    RunState,
)
from deep_research.persistence.memory import InMemoryRunRepository
from deep_research.persistence.runs import ConcurrencyConflictError, RunNotFoundError
from deep_research.services.runs import (
    IdempotencyConflictError,
    InvalidRunTransitionError,
    RunControlService,
    StalePlanApprovalError,
)


@pytest.fixture
def principal() -> Principal:
    return Principal(subject="user-1", tenant_id="tenant-1")


@pytest.fixture
def repository() -> InMemoryRunRepository:
    return InMemoryRunRepository()


@pytest.fixture
def service(repository: InMemoryRunRepository) -> RunControlService:
    return RunControlService(repository)


async def _create(service: RunControlService, principal: Principal, *, key: str = "create-key"):
    return await service.create_run(
        principal,
        CreateRunRequest(topic="How is grid storage changing?"),
        idempotency_key=key,
    )


def _plan(brief: ResearchBrief, version: int = 1) -> ResearchPlan:
    draft = ResearchPlanDraft(
        questions=[
            ResearchQuestion(
                id="market_change",
                question="What has changed?",
                success_criteria=["Identify material changes"],
            )
        ],
        workstreams=[
            Workstream(
                id="market",
                title="Market",
                objective="Measure changes",
                research_question_ids=["market_change"],
                source_priorities=["Primary sources"],
            )
        ],
        source_strategy=SourceStrategy(
            preferred_source_types=["Official publications"],
            freshness_requirements="Prefer the latest available data",
            corroboration_rules=["Corroborate material claims"],
        ),
        outline=[
            ReportSection(
                id="findings",
                title="Findings",
                purpose="Answer the research question",
                research_question_ids=["market_change"],
            )
        ],
    )
    return ResearchPlan.finalize(
        draft=draft,
        brief=brief,
        budget=BudgetLimits.for_preset(DepthPreset.STANDARD),
        version=version,
    )


@pytest.mark.asyncio
async def test_complete_run_lifecycle_is_persisted_as_ordered_events(
    service: RunControlService, principal: Principal
) -> None:
    run = await _create(service, principal)
    run = await service.start_run(principal, run.run_id, idempotency_key="start-key")
    brief = ResearchBrief(topic=run.topic, objective="Explain market changes")
    run = await service.record_clarification(
        principal,
        run.run_id,
        brief,
        scope_ready=True,
        idempotency_key="brief-key",
    )
    plan = _plan(brief)
    run = await service.save_plan(principal, run.run_id, plan, idempotency_key="plan-save")
    run = await service.approve_plan(
        principal,
        run.run_id,
        PlanApprovalRequest(version=plan.version, content_hash=plan.content_hash),
        idempotency_key="plan-okay",
    )
    for index, state in enumerate(
        [
            RunState.REVIEWING,
            RunState.GENERATING_REPORT,
            RunState.GENERATING_QUESTIONS,
            RunState.COMPLETED,
        ]
    ):
        run = await service.advance(
            principal, run.run_id, state, idempotency_key=f"advance-{index}"
        )

    assert run.state is RunState.COMPLETED
    assert run.approved_plan_hash == plan.content_hash
    page = await service.get_events(principal, run.run_id)
    assert [event.cursor for event in page.events] == list(range(1, 10))
    assert page.next_cursor == 9
    replay = await service.get_events(principal, run.run_id, after_cursor=7)
    assert [event.cursor for event in replay.events] == [8, 9]


@pytest.mark.asyncio
async def test_idempotency_replays_response_and_rejects_a_different_command(
    service: RunControlService, principal: Principal
) -> None:
    first = await _create(service, principal)
    replay = await _create(service, principal)

    assert replay == first
    with pytest.raises(IdempotencyConflictError):
        await service.create_run(
            principal,
            CreateRunRequest(topic="A different topic"),
            idempotency_key="create-key",
        )


@pytest.mark.asyncio
async def test_idempotency_key_cannot_replay_a_command_for_another_run(
    service: RunControlService, principal: Principal
) -> None:
    first = await _create(service, principal, key="create-one")
    second = await _create(service, principal, key="create-two")
    await service.start_run(principal, first.run_id, idempotency_key="shared-command-key")

    with pytest.raises(IdempotencyConflictError):
        await service.start_run(principal, second.run_id, idempotency_key="shared-command-key")


@pytest.mark.asyncio
async def test_stale_plan_approval_and_terminal_transitions_are_rejected(
    service: RunControlService, principal: Principal
) -> None:
    run = await _create(service, principal)
    run = await service.start_run(principal, run.run_id, idempotency_key="start-key")
    brief = ResearchBrief(topic=run.topic)
    run = await service.record_clarification(
        principal,
        run.run_id,
        brief,
        scope_ready=True,
        idempotency_key="brief-key",
    )
    plan = _plan(brief)
    run = await service.save_plan(principal, run.run_id, plan, idempotency_key="plan-save")

    with pytest.raises(StalePlanApprovalError):
        await service.approve_plan(
            principal,
            run.run_id,
            PlanApprovalRequest(version=plan.version, content_hash="0" * 64),
            idempotency_key="stale-key",
        )

    run = await service.cancel(principal, run.run_id, idempotency_key="cancel-key")
    with pytest.raises(InvalidRunTransitionError):
        await service.start_run(principal, run.run_id, idempotency_key="restart-key")


@pytest.mark.asyncio
async def test_repository_enforces_owner_and_optimistic_revision(
    repository: InMemoryRunRepository,
    service: RunControlService,
    principal: Principal,
) -> None:
    run = await _create(service, principal)
    other = Principal(subject="user-2", tenant_id=principal.tenant_id)
    with pytest.raises(RunNotFoundError):
        await repository.get(other, run.run_id)

    first, second = await asyncio.gather(
        repository.get(principal, run.run_id),
        repository.get(principal, run.run_id),
    )
    first_update, first_event = service._mutation(  # noqa: SLF001
        first,
        state=RunState.CLARIFYING,
        event_type="run.started",
        event_payload={"state": RunState.CLARIFYING.value},
    )
    second_update, second_event = service._mutation(  # noqa: SLF001
        second,
        state=RunState.CLARIFYING,
        event_type="run.started",
        event_payload={"state": RunState.CLARIFYING.value},
    )
    first_idem = service._idempotency(  # noqa: SLF001
        principal, "first-key", "1" * 64, first_update
    )
    second_idem = service._idempotency(  # noqa: SLF001
        principal, "second-key", "2" * 64, second_update
    )
    await repository.commit(
        first_update, first_event, expected_revision=first.revision, idempotency=first_idem
    )
    with pytest.raises(ConcurrencyConflictError):
        await repository.commit(
            second_update,
            second_event,
            expected_revision=second.revision,
            idempotency=second_idem,
        )


@pytest.mark.asyncio
async def test_budget_usage_must_be_monotonic(
    service: RunControlService, principal: Principal
) -> None:
    run = await _create(service, principal)
    run = await service.record_budget(
        principal,
        run.run_id,
        BudgetUsage(searches=2, model_calls=1),
        idempotency_key="budget-one",
    )
    with pytest.raises(ValueError, match="cannot decrease"):
        await service.record_budget(
            principal,
            run.run_id,
            BudgetUsage(searches=1, model_calls=1),
            idempotency_key="budget-two",
        )


@pytest.mark.asyncio
async def test_expired_local_idempotency_record_is_not_replayed(
    repository: InMemoryRunRepository,
) -> None:
    now = datetime.now(UTC)
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    service = RunControlService(repository, retention_days=1, clock=lambda: now)
    run = await _create(service, principal)
    repository._idempotency[
        (  # noqa: SLF001
            principal.tenant_id,
            principal.subject,
            "create-key",
        )
    ] = repository._idempotency[
        (  # noqa: SLF001
            principal.tenant_id,
            principal.subject,
            "create-key",
        )
    ].model_copy(update={"expires_at": now})

    assert await repository.get_idempotency(principal, "create-key") is None
    assert run.expires_at > now
