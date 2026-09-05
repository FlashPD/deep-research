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
    assert plan.budget.max_search_queries == 5
    assert plan.content_hash == plan.calculate_hash()
    assert plan.diagram_candidates == []
    assert gateway.calls[0]["role"] == "planner"
    assert "no web" in gateway.calls[0]["system_prompt"]
    assert '"max_initial_search_queries": 4' in gateway.calls[0]["prompt"]
    assert "single-purpose" in gateway.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_quick_plan_keeps_explicitly_requested_table() -> None:
    draft = make_plan_draft()
    plan = await PlanningAgent(FakeGateway(draft)).create_plan(
        PlannerRequest(
            brief=ResearchBrief(
                topic="EV market",
                output_expectations=["Include a comparison table."],
            ),
            depth=DepthPreset.QUICK,
        )
    )

    assert plan.diagram_candidates == draft.diagram_candidates


@pytest.mark.asyncio
async def test_planner_repairs_draft_that_exceeds_search_ceiling() -> None:
    gateway = FakeGateway(make_plan_draft(query_count=6), make_plan_draft(query_count=2))
    planner = PlanningAgent(gateway)

    plan = await planner.create_plan(
        PlannerRequest(brief=ResearchBrief(topic="EV market"), depth=DepthPreset.QUICK)
    )

    assert len(gateway.calls) == 2
    assert "failed deterministic validation" in gateway.calls[1]["prompt"]
    assert plan.version == 1


@pytest.mark.asyncio
async def test_planner_repairs_overly_detailed_quick_plan() -> None:
    base = make_plan_draft(query_count=2)
    extra_questions = [
        base.questions[0].model_copy(update={"id": "market_growth"}),
        base.questions[0].model_copy(update={"id": "market_outlook"}),
    ]
    question_ids = ["market_size", "market_growth", "market_outlook"]
    oversized = base.model_copy(
        update={
            "questions": [*base.questions, *extra_questions],
            "workstreams": [
                base.workstreams[0].model_copy(
                    update={"research_question_ids": question_ids}
                )
            ],
            "outline": [
                base.outline[0].model_copy(
                    update={"research_question_ids": question_ids}
                )
            ],
        }
    )
    gateway = FakeGateway(oversized, base)

    plan = await PlanningAgent(gateway).create_plan(
        PlannerRequest(brief=ResearchBrief(topic="EV market"), depth=DepthPreset.QUICK)
    )

    assert len(gateway.calls) == 2
    assert "quick plan has 3 questions" in gateway.calls[1]["prompt"]
    assert len(plan.questions) == 1


@pytest.mark.asyncio
async def test_quick_plan_reserves_one_search_for_repair() -> None:
    gateway = FakeGateway(make_plan_draft(query_count=5), make_plan_draft(query_count=4))

    plan = await PlanningAgent(gateway).create_plan(
        PlannerRequest(brief=ResearchBrief(topic="EV market"), depth=DepthPreset.QUICK)
    )

    assert len(gateway.calls) == 2
    assert "initial ceiling 4" in gateway.calls[1]["prompt"]
    assert len(plan.workstreams[0].candidate_queries) == 4


@pytest.mark.asyncio
async def test_quick_plan_removes_artificial_workstream_dependencies() -> None:
    base = make_plan_draft(query_count=1)
    second_question = base.questions[0].model_copy(update={"id": "market_growth"})
    dependent = base.workstreams[0].model_copy(
        update={
            "id": "growth_analysis",
            "research_question_ids": ["market_growth"],
            "dependencies": ["market_analysis"],
        }
    )
    with_dependency = base.model_copy(
        update={
            "questions": [*base.questions, second_question],
            "workstreams": [*base.workstreams, dependent],
            "outline": [
                base.outline[0].model_copy(
                    update={"research_question_ids": ["market_size", "market_growth"]}
                )
            ],
        }
    )
    repaired = with_dependency.model_copy(
        update={
            "workstreams": [
                item.model_copy(update={"dependencies": []})
                for item in with_dependency.workstreams
            ]
        }
    )
    gateway = FakeGateway(with_dependency, repaired)

    plan = await PlanningAgent(gateway).create_plan(
        PlannerRequest(brief=ResearchBrief(topic="EV market"), depth=DepthPreset.QUICK)
    )

    assert len(gateway.calls) == 2
    assert "must be independent" in gateway.calls[1]["prompt"]
    assert all(not item.dependencies for item in plan.workstreams)


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
