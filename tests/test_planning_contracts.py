import pytest
from pydantic import ValidationError

from deep_research.contracts.clarification import ResearchBrief
from deep_research.contracts.planning import BudgetLimits, DepthPreset, ResearchPlan
from tests.factories import make_plan_draft


def test_depth_presets_match_architecture_ceilings() -> None:
    deep = BudgetLimits.for_preset(DepthPreset.DEEP)

    assert deep.target_duration_seconds == 2_700
    assert deep.max_search_queries == 50
    assert deep.max_accepted_sources == 75
    assert deep.research_concurrency == 10
    assert deep.reviewer_retries == 2


def test_finalized_plan_has_verifiable_content_hash() -> None:
    plan = ResearchPlan.finalize(
        draft=make_plan_draft(),
        brief=ResearchBrief(topic="EV market"),
        budget=BudgetLimits.for_preset(DepthPreset.STANDARD),
        version=1,
    )

    assert plan.content_hash == plan.calculate_hash()

    tampered = plan.model_dump(mode="json")
    tampered["brief"]["topic"] = "Tampered topic"
    with pytest.raises(ValidationError, match="content_hash"):
        ResearchPlan.model_validate(tampered)


def test_workstream_cycle_is_rejected() -> None:
    draft = make_plan_draft().model_dump(mode="json")
    draft["workstreams"][0]["dependencies"] = ["market_analysis"]

    with pytest.raises(ValidationError, match="cannot depend on itself"):
        type(make_plan_draft()).model_validate(draft)


def test_every_research_question_must_appear_in_report_outline() -> None:
    draft = make_plan_draft().model_dump(mode="json")
    draft["outline"][0]["research_question_ids"] = []

    with pytest.raises(ValidationError, match="not represented in the report outline"):
        type(make_plan_draft()).model_validate(draft)
