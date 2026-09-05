import pytest
from pydantic import ValidationError

from deep_research.contracts.clarification import (
    AnswerType,
    ClarificationDecision,
    ClarificationQuestion,
    ResearchBrief,
)


def test_select_question_requires_options() -> None:
    with pytest.raises(ValidationError):
        ClarificationQuestion(
            id="audience",
            question="Who is the audience?",
            rationale="This controls depth.",
            expected_answer_type=AnswerType.SINGLE_SELECT,
        )


def test_scope_ready_rejects_questions() -> None:
    with pytest.raises(ValidationError):
        ClarificationDecision(
            status="scope_ready",
            brief=ResearchBrief(topic="Battery recycling"),
            interpretation_summary="Scope is sufficient.",
            questions=[
                ClarificationQuestion(
                    id="audience",
                    question="Who is the audience?",
                    rationale="This controls depth.",
                    expected_answer_type=AnswerType.TEXT,
                )
            ],
        )
