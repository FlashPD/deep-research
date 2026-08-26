import pytest

from deep_research.agents.clarifier import ClarifierAgent
from deep_research.contracts.clarification import (
    AnswerType,
    ClarificationAnswer,
    ClarificationDecision,
    ClarificationQuestion,
    ClarifierRequest,
    ResearchBrief,
)
from tests.conftest import FakeGateway


def decision_needing_detail() -> ClarificationDecision:
    return ClarificationDecision(
        status="needs_clarification",
        brief=ResearchBrief(topic="EV market"),
        interpretation_summary="Research the EV market, with scope still unresolved.",
        questions=[
            ClarificationQuestion(
                id="geography",
                question="Which geography should the analysis cover?",
                rationale="Market conditions differ materially by geography.",
                expected_answer_type=AnswerType.TEXT,
            )
        ],
    )


@pytest.mark.asyncio
async def test_clarifier_has_no_tools_and_returns_typed_decision() -> None:
    gateway = FakeGateway(decision_needing_detail())
    agent = ClarifierAgent(gateway)

    result = await agent.evaluate(ClarifierRequest(topic="EV market"))

    assert result.status == "needs_clarification"
    assert gateway.calls[0]["role"] == "clarifier"
    assert "no tools" in gateway.calls[0]["system_prompt"]


@pytest.mark.asyncio
async def test_third_round_forces_explicit_confirmation() -> None:
    gateway = FakeGateway(decision_needing_detail())
    agent = ClarifierAgent(gateway)

    result = await agent.evaluate(ClarifierRequest(topic="EV market", round_number=3))

    assert len(result.questions) == 1
    assert result.questions[0].id == "confirm_best_interpretation"
    assert result.questions[0].expected_answer_type is AnswerType.CONFIRMATION


@pytest.mark.asyncio
async def test_confirmation_resumes_without_a_fourth_model_round() -> None:
    gateway = FakeGateway()
    agent = ClarifierAgent(gateway)
    brief = ResearchBrief(topic="EV market", geography=["United States"])

    result = await agent.evaluate(
        ClarifierRequest(
            topic="EV market",
            round_number=3,
            current_brief=brief,
            answers=[
                ClarificationAnswer(
                    question_id="confirm_best_interpretation",
                    value=True,
                )
            ],
        )
    )

    assert result.status == "scope_ready"
    assert result.brief == brief
    assert gateway.calls == []
