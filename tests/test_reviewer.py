import pytest

from deep_research.agents.reviewer import EvidenceReviewer
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
