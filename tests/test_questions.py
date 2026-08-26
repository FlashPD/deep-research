import pytest

from deep_research.agents.questions import QuestionsAgent
from deep_research.contracts.questions import QuestionsRequest
from tests.conftest import FakeGateway
from tests.factories import make_question_set, make_report_context


@pytest.mark.asyncio
async def test_questions_agent_returns_prioritized_report_linked_questions() -> None:
    gateway = FakeGateway(make_question_set())
    agent = QuestionsAgent(gateway)

    result = await agent.generate(QuestionsRequest(report=make_report_context(), count=5))

    assert [question.priority for question in result.questions] == [1, 2, 3, 4, 5]
    assert gateway.calls[0]["role"] == "questions"
    assert "no tools" in gateway.calls[0]["system_prompt"]


@pytest.mark.asyncio
async def test_questions_agent_repairs_unknown_section_reference() -> None:
    gateway = FakeGateway(
        make_question_set(originating_section_id="unknown_section"),
        make_question_set(),
    )
    agent = QuestionsAgent(gateway)

    result = await agent.generate(QuestionsRequest(report=make_report_context(), count=5))

    assert len(gateway.calls) == 2
    assert "unknown report sections" in gateway.calls[1]["prompt"]
    assert len(result.questions) == 5
