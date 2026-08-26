import pytest

from deep_research.agents.planner import PlanningAgent
from deep_research.contracts.clarification import ResearchBrief
from deep_research.contracts.planning import DepthPreset, PlannerRequest
from tests.conftest import FakeGateway
from tests.factories import make_plan_draft


@pytest.mark.asyncio
async def test_planner_enforces_budget_and_finalizes_control_fields() -> None:
    gateway = FakeGateway(make_plan_draft())
    planner = PlanningAgent(gateway)

    plan = await planner.create_plan(
        PlannerRequest(brief=ResearchBrief(topic="EV market"), depth=DepthPreset.QUICK)
    )

    assert plan.version == 1
    assert plan.budget.max_search_queries == 8
    assert plan.content_hash == plan.calculate_hash()
    assert gateway.calls[0]["role"] == "planner"
    assert "no web" in gateway.calls[0]["system_prompt"]


@pytest.mark.asyncio
async def test_planner_repairs_draft_that_exceeds_search_ceiling() -> None:
    gateway = FakeGateway(make_plan_draft(query_count=9), make_plan_draft(query_count=2))
    planner = PlanningAgent(gateway)

    plan = await planner.create_plan(
        PlannerRequest(brief=ResearchBrief(topic="EV market"), depth=DepthPreset.QUICK)
    )

    assert len(gateway.calls) == 2
    assert "failed deterministic validation" in gateway.calls[1]["prompt"]
    assert plan.version == 1


@pytest.mark.asyncio
async def test_regenerated_plan_increments_version() -> None:
    first_gateway = FakeGateway(make_plan_draft())
    first = await PlanningAgent(first_gateway).create_plan(
        PlannerRequest(brief=ResearchBrief(topic="EV market"))
    )
    second_gateway = FakeGateway(make_plan_draft())

    second = await PlanningAgent(second_gateway).create_plan(
        PlannerRequest(
            brief=ResearchBrief(topic="EV market"),
            previous_plan=first,
            user_edits=["Emphasize charging infrastructure."],
        )
    )

    assert second.version == 2
    assert second.content_hash != first.content_hash
