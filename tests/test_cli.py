from argparse import Namespace

import pytest

from deep_research.cli import _prompt_for_answer, run_cli
from deep_research.client.polling import RunEventPoller
from deep_research.contracts.clarification import (
    AnswerType,
    ClarificationQuestion,
)
from deep_research.contracts.runs import Principal, RunState
from deep_research.dev_app import LocalDevelopmentRuntime
from deep_research.jobs.memory import InMemoryJobDispatcher
from deep_research.persistence.memory import InMemoryRunRepository
from deep_research.services.runs import RunControlService
from deep_research.settings import AppSettings
from deep_research.worker import DurableWorker, WorkerAgents
from tests.conftest import FakeGateway
from tests.test_clarification_resume import RecordingClarifier
from tests.test_worker import _agents


def test_cli_collects_typed_multi_select_answers() -> None:
    output: list[str] = []
    question = ClarificationQuestion(
        id="regions",
        question="Which regions?",
        rationale="This bounds comparison.",
        expected_answer_type=AnswerType.MULTI_SELECT,
        options=["Americas", "Europe", "Asia"],
    )

    answer = _prompt_for_answer(
        question, input_fn=lambda _prompt: "1,3", output_fn=output.append
    )

    assert answer == ["Americas", "Asia"]


@pytest.mark.asyncio
async def test_event_poller_advances_its_cursor() -> None:
    from deep_research.contracts.runs import CreateRunRequest, Principal

    principal = Principal(subject="user", tenant_id="tenant")
    control = RunControlService(InMemoryRunRepository())
    run = await control.create_run(
        principal, CreateRunRequest(topic="Polling"), idempotency_key="poll-create"
    )
    await control.start_run(principal, run.run_id, idempotency_key="poll-start")
    poller = RunEventPoller()

    first = await poller.poll(control, principal, run.run_id)
    second = await poller.poll(control, principal, run.run_id)

    assert [event.event_type for event in first] == ["run.created", "run.started"]
    assert second == []
    assert poller.cursor == 2


@pytest.mark.asyncio
async def test_cli_completes_mocked_workflow_through_worker_boundaries(tmp_path) -> None:
    repository = InMemoryRunRepository()
    dispatcher = InMemoryJobDispatcher()
    control = RunControlService(repository)
    base_agents = _agents([])
    worker = DurableWorker(
        control,
        dispatcher,
        WorkerAgents(
            clarifier=RecordingClarifier(),
            planner=base_agents.planner,
            researcher_factory=base_agents.researcher_factory,
            reviewer=base_agents.reviewer,
            report=base_agents.report,
            questions=base_agents.questions,
        ),
    )
    runtime = LocalDevelopmentRuntime(
        settings=AppSettings(),
        repository=repository,
        dispatcher=dispatcher,
        control=control,
        gateway=FakeGateway(),
        worker=worker,
    )
    responses = iter(["Engineering leaders", "y"])
    output: list[str] = []
    args = Namespace(
        topic="Ambiguous EV market",
        depth="quick",
        provider=None,
        output=tmp_path,
        auto_approve_plan=False,
    )

    exit_code = await run_cli(
        args,
        runtime,
        input_fn=lambda _prompt: next(responses),
        output_fn=output.append,
    )

    reports = list(tmp_path.glob("*.md"))
    assert exit_code == 0
    assert len(reports) == 1
    assert "# EV Market Report" in reports[0].read_text(encoding="utf-8")
    assert any("Generated research plan" in line for line in output)
    assert any("Follow-up questions" in line for line in output)


@pytest.mark.asyncio
async def test_cli_keyboard_interrupt_cooperatively_cancels_run(tmp_path) -> None:
    repository = InMemoryRunRepository()
    dispatcher = InMemoryJobDispatcher()
    control = RunControlService(repository)
    base_agents = _agents([])
    worker = DurableWorker(
        control,
        dispatcher,
        WorkerAgents(
            clarifier=RecordingClarifier(),
            planner=base_agents.planner,
            researcher_factory=base_agents.researcher_factory,
            reviewer=base_agents.reviewer,
            report=base_agents.report,
            questions=base_agents.questions,
        ),
    )
    runtime = LocalDevelopmentRuntime(
        settings=AppSettings(),
        repository=repository,
        dispatcher=dispatcher,
        control=control,
        gateway=FakeGateway(),
        worker=worker,
    )
    output: list[str] = []
    args = Namespace(
        topic="Interrupt this run",
        depth="quick",
        provider=None,
        output=tmp_path,
        auto_approve_plan=False,
    )

    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    exit_code = await run_cli(args, runtime, input_fn=interrupt, output_fn=output.append)
    run_id = output[0].split()[1]
    cancelled = await control.get_run(
        Principal(subject="local-user", tenant_id="local-tenant"), run_id
    )

    assert exit_code == 130
    assert cancelled.state is RunState.CANCELLED
    assert any("Cancellation requested" in line for line in output)
