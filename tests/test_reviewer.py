import pytest

from deep_research.agents.reviewer import EvidenceReviewer, compact_limitations
from deep_research.contracts.evidence import (
    BudgetUsage,
    CoverageStatus,
    ReviewerRequest,
    ReviewState,
)
from deep_research.contracts.planning import DepthPreset
from tests.conftest import FakeGateway
from tests.factories import make_evidence_package, make_plan, make_review_draft


@pytest.mark.asyncio
async def test_reviewer_approves_complete_evidence_and_calculates_scores() -> None:
    gateway = FakeGateway(make_review_draft())

    result = await EvidenceReviewer(gateway).review(
        ReviewerRequest(plan=make_plan(), evidence=make_evidence_package())
    )

    assert result.review_state is ReviewState.APPROVED
    assert result.coverage_score == 1
    assert result.citation_coverage == 1
    assert gateway.calls[0]["role"] == "reviewer"
    assert "no research tools" in gateway.calls[0]["system_prompt"]
    assert "Do not emit analysis" in gateway.calls[0]["system_prompt"]
    assert "source_scores_exact_ids" in gateway.calls[0]["prompt"]
    assert "Compact review input" in gateway.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_reviewer_requests_targeted_repair_within_ceiling() -> None:
    gateway = FakeGateway(
        make_review_draft(
            coverage=CoverageStatus.PARTIAL,
            recommends_approval=False,
            with_retry=True,
        )
    )

    result = await EvidenceReviewer(gateway).review(
        ReviewerRequest(plan=make_plan(), evidence=make_evidence_package())
    )

    assert result.review_state is ReviewState.REPAIR_REQUIRED
    assert result.retry_tasks[0].research_question_ids == ["market_size"]


@pytest.mark.asyncio
async def test_quick_reviewer_allows_adaptive_sixth_query_for_two_material_gaps() -> None:
    draft = make_review_draft(
        coverage=CoverageStatus.PARTIAL,
        recommends_approval=False,
        with_retry=True,
    )
    first = draft.retry_tasks[0]
    second = first.model_copy(
        update={
            "task_id": "repair_current_price",
            "objective": "Confirm current retail pricing from a major retailer.",
            "candidate_queries": ["current EV retail market price"],
        }
    )
    draft = draft.model_copy(update={"retry_tasks": [first, second]})
    plan = make_plan(preset=DepthPreset.QUICK)

    result = await EvidenceReviewer(FakeGateway(draft)).review(
        ReviewerRequest(
            plan=plan,
            evidence=make_evidence_package(),
            budget_usage=BudgetUsage(searches=4),
        )
    )

    assert result.review_state is ReviewState.REPAIR_REQUIRED
    assert sum(len(task.candidate_queries) for task in result.retry_tasks) == 2


def test_compact_limitations_deduplicates_and_discloses_omissions() -> None:
    limitations = [
        "No measured review was captured.",
        "No measured review was captured.",
        "Current pricing was not confirmed.",
        "Audio testing was unavailable.",
        "Regional panel variation remains unresolved.",
    ]

    compact = compact_limitations(limitations, max_items=3)

    assert compact[0] == "No measured review was captured."
    assert len(compact) == 3
    assert "omitted" in compact[-1]


@pytest.mark.asyncio
async def test_reviewer_records_limitations_when_budget_is_exhausted() -> None:
    gateway = FakeGateway(
        make_review_draft(
            coverage=CoverageStatus.PARTIAL,
            recommends_approval=False,
        )
    )
    plan = make_plan(preset=DepthPreset.QUICK)

    result = await EvidenceReviewer(gateway).review(
        ReviewerRequest(
            plan=plan,
            evidence=make_evidence_package(),
            budget_usage=BudgetUsage(searches=plan.budget.max_search_queries),
        )
    )

    assert result.review_state is ReviewState.APPROVED_WITH_LIMITATIONS
    assert result.retry_tasks == []
    assert any("budget was exhausted" in item for item in result.limitations)


@pytest.mark.asyncio
async def test_reviewer_cannot_approve_coverage_without_evidence_backed_claims() -> None:
    draft = make_review_draft(with_retry=True).model_copy(update={"source_scores": []})
    gateway = FakeGateway(draft)

    result = await EvidenceReviewer(gateway).review(
        ReviewerRequest(
            plan=make_plan(),
            evidence=make_evidence_package().model_copy(
                update={"sources": [], "excerpts": [], "claims": []}
            ),
        )
    )

    assert result.review_state is ReviewState.REPAIR_REQUIRED
    assert any("no evidence-backed claim" in item for item in result.deterministic_issues)


@pytest.mark.asyncio
async def test_reviewer_rejects_empty_evidence_when_budget_is_exhausted() -> None:
    draft = make_review_draft(
        coverage=CoverageStatus.MISSING,
        recommends_approval=False,
    ).model_copy(update={"source_scores": []})
    gateway = FakeGateway(draft)
    plan = make_plan(preset=DepthPreset.QUICK)

    result = await EvidenceReviewer(gateway).review(
        ReviewerRequest(
            plan=plan,
            evidence=make_evidence_package().model_copy(
                update={"sources": [], "excerpts": [], "claims": []}
            ),
            budget_usage=BudgetUsage(searches=plan.budget.max_search_queries),
        )
    )

    assert result.review_state is ReviewState.REJECTED
    assert result.retry_tasks == []


def test_reviewer_fallback_allows_only_deterministically_complete_evidence() -> None:
    request = ReviewerRequest(plan=make_plan(), evidence=make_evidence_package())

    result = EvidenceReviewer.deterministic_fallback(
        request, RuntimeError("structured review unavailable")
    )

    assert result.review_state is ReviewState.APPROVED_WITH_LIMITATIONS
    assert result.recommends_approval is False
    assert result.deterministic_issues == []
    assert any("partial report" in item for item in result.limitations)
    assert all(score.authority == 0.5 for score in result.source_scores)


@pytest.mark.asyncio
async def test_gap_without_retry_task_is_approved_with_limitations_not_an_error() -> None:
    # The model judged the remaining gap unrepairable (no retry task) while retries remain.
    draft = make_review_draft(coverage=CoverageStatus.PARTIAL, recommends_approval=False)
    assert draft.retry_tasks == []
    gateway = FakeGateway(draft)

    result = await EvidenceReviewer(gateway).review(
        ReviewerRequest(plan=make_plan(), evidence=make_evidence_package())
    )

    assert len(gateway.calls) == 1
    assert result.review_state is ReviewState.APPROVED_WITH_LIMITATIONS
    assert result.retry_tasks == []
    assert any("proposed no repair task" in item for item in result.limitations)


def _plan_with_narrative_section():
    from deep_research.contracts.clarification import ResearchBrief
    from deep_research.contracts.planning import BudgetLimits, ReportSection, ResearchPlan
    from tests.factories import make_plan_draft

    draft = make_plan_draft()
    draft = draft.model_copy(
        update={
            "outline": [
                ReportSection(
                    id="intro", title="Introduction & Scope", purpose="Frame the report."
                ),
                *draft.outline,
            ]
        }
    )
    return ResearchPlan.finalize(
        draft=draft,
        brief=ResearchBrief(topic="EV market"),
        budget=BudgetLimits.for_preset(DepthPreset.DEEP),
        version=1,
    )


def test_fallback_treats_narrative_sections_as_limitations_not_rejection() -> None:
    request = ReviewerRequest(plan=_plan_with_narrative_section(), evidence=make_evidence_package())

    result = EvidenceReviewer.deterministic_fallback(
        request, RuntimeError("structured review unavailable")
    )

    assert result.review_state is ReviewState.APPROVED_WITH_LIMITATIONS
    assert not any("intro" in item for item in result.deterministic_issues)
    assert [item.item_id for item in result.section_coverage] == ["intro", "market_findings"]


def test_fallback_still_rejects_structurally_corrupt_evidence() -> None:
    evidence = make_evidence_package()
    corrupt = evidence.model_copy(
        update={
            "claims": [
                evidence.claims[0].model_copy(update={"research_question_id": "not_in_plan"})
            ]
        }
    )
    request = ReviewerRequest(plan=make_plan(), evidence=corrupt)

    result = EvidenceReviewer.deterministic_fallback(
        request, RuntimeError("structured review unavailable")
    )

    assert result.review_state is ReviewState.REJECTED
    assert any("unknown research question" in item for item in result.deterministic_issues)


def test_reviewer_fallback_rejects_deterministically_incomplete_evidence() -> None:
    request = ReviewerRequest(
        plan=make_plan(),
        evidence=make_evidence_package().model_copy(
            update={"sources": [], "excerpts": [], "claims": []}
        ),
    )

    result = EvidenceReviewer.deterministic_fallback(
        request, RuntimeError("structured review unavailable")
    )

    assert result.review_state is ReviewState.REJECTED
    assert result.deterministic_issues
