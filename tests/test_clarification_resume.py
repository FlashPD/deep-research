import pytest

from deep_research.contracts.clarification import (
    AnswerType,
    ClarificationAnswer,
    ClarificationDecision,
    ClarificationQuestion,
    ResearchBrief,
)
from deep_research.contracts.jobs import JobPhase
from deep_research.contracts.runs import CreateRunRequest, Principal, RunState
from deep_research.jobs.memory import InMemoryJobDispatcher
from deep_research.persistence.memory import InMemoryRunRepository
from deep_research.services.runs import RunCommandConflictError, RunControlService
from deep_research.worker import DurableWorker, WorkerAgents, make_phase_job
from tests.test_worker import _agents


class RecordingClarifier:
    def __init__(self) -> None:
        self.requests = []

    async def evaluate(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return ClarificationDecision(
                status="needs_clarification",
                brief=ResearchBrief(topic=request.topic, audience="Proposed audience"),
                questions=[
                    ClarificationQuestion(
                        id="audience",
                        question="Who is the audience?",
                        rationale="This changes the report framing.",
                        expected_answer_type=AnswerType.TEXT,
                    )
                ],
                interpretation_summary="Audience is still required.",
            )
        return ClarificationDecision(
            status="scope_ready",
            brief=ResearchBrief(topic=request.topic, audience=str(request.answers[0].value)),
            interpretation_summary="The answer completed the scope.",
        )


@pytest.mark.asyncio
async def test_clarification_answers_resume_from_checkpoint_and_replay_idempotently() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    clarifier = RecordingClarifier()
    agents = _agents([])
    worker = DurableWorker(
        control,
        dispatcher,
        WorkerAgents(
            clarifier=clarifier,
            planner=agents.planner,
            researcher_factory=agents.researcher_factory,
            reviewer=agents.reviewer,
            report=agents.report,
            questions=agents.questions,
        ),
    )
    created = await control.create_run(
        principal, CreateRunRequest(topic="Ambiguous topic"), idempotency_key="create-run"
    )
    run = await control.start_run(principal, created.run_id, idempotency_key="start-run")
    await worker.handle(make_phase_job(run, JobPhase.CLARIFY))
    interrupted = await control.get_run(principal, run.run_id)
    assert interrupted.graph_checkpoint.clarification_round == 1
    assert interrupted.graph_checkpoint.pending_clarification_questions[0].id == "audience"

    answers = [ClarificationAnswer(question_id="audience", value="Engineering leaders")]
    resumed = await control.submit_clarification_answers(
        principal,
        run.run_id,
        answers,
        round_number=1,
        idempotency_key="answer-round-one",
    )
    replay = await control.submit_clarification_answers(
        principal,
        run.run_id,
        answers,
        round_number=1,
        idempotency_key="answer-round-one",
    )
    assert replay.revision == resumed.revision

    await worker.handle(make_phase_job(resumed, JobPhase.CLARIFY))
    completed = await control.get_run(principal, run.run_id)
    assert completed.state is RunState.PLANNING
    assert clarifier.requests[1].round_number == 2
    assert clarifier.requests[1].answers == answers
    assert clarifier.requests[1].current_brief == interrupted.brief
    assert completed.graph_checkpoint.clarification_answers == answers


@pytest.mark.asyncio
async def test_clarification_rejects_stale_unknown_and_missing_required_answers() -> None:
    principal = Principal(subject="user-1", tenant_id="tenant-1")
    control = RunControlService(InMemoryRunRepository())
    dispatcher = InMemoryJobDispatcher()
    clarifier = RecordingClarifier()
    agents = _agents([])
    worker = DurableWorker(
        control,
        dispatcher,
        WorkerAgents(
            clarifier=clarifier,
            planner=agents.planner,
            researcher_factory=agents.researcher_factory,
            reviewer=agents.reviewer,
            report=agents.report,
            questions=agents.questions,
        ),
    )
    created = await control.create_run(
        principal, CreateRunRequest(topic="Ambiguous topic"), idempotency_key="create-run"
    )
    run = await control.start_run(principal, created.run_id, idempotency_key="start-run")
    await worker.handle(make_phase_job(run, JobPhase.CLARIFY))

    with pytest.raises(RunCommandConflictError, match="stale round"):
        await control.submit_clarification_answers(
            principal,
            run.run_id,
            [ClarificationAnswer(question_id="audience", value="Leaders")],
            round_number=2,
            idempotency_key="stale-answer",
        )
    with pytest.raises(RunCommandConflictError, match="unknown or stale"):
        await control.submit_clarification_answers(
            principal,
            run.run_id,
            [ClarificationAnswer(question_id="unknown", value="Leaders")],
            round_number=1,
            idempotency_key="unknown-answer",
        )
    with pytest.raises(RunCommandConflictError, match="required clarification"):
        await control.submit_clarification_answers(
            principal,
            run.run_id,
            [],
            round_number=1,
            idempotency_key="missing-answer",
        )
