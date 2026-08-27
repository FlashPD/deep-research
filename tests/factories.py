from deep_research.contracts.clarification import ResearchBrief
from deep_research.contracts.evidence import (
    CoverageAssessment,
    CoverageStatus,
    EvidenceClaim,
    EvidenceExcerpt,
    EvidencePackage,
    ReviewDraft,
    SourceQualityScore,
    SourceRecord,
    SourceType,
    SupportStrength,
)
from deep_research.contracts.planning import (
    BudgetLimits,
    DepthPreset,
    DiagramCandidate,
    ReportSection,
    ResearchPlanDraft,
    ResearchQuestion,
    SourceStrategy,
    Workstream,
)
from deep_research.contracts.questions import (
    FollowUpQuestion,
    FollowUpQuestionSet,
    ReportContext,
    ReportSectionSummary,
)
from deep_research.contracts.reporting import ReportDraft, ReportSectionDraft


def make_plan_draft(*, uses_uploads: bool = False, query_count: int = 2) -> ResearchPlanDraft:
    return ResearchPlanDraft(
        questions=[
            ResearchQuestion(
                id="market_size",
                question="How large is the market?",
                success_criteria=["Quantify the current market using primary evidence."],
            )
        ],
        workstreams=[
            Workstream(
                id="market_analysis",
                title="Market analysis",
                objective="Measure the market and its direction.",
                research_question_ids=["market_size"],
                candidate_queries=[f"market query {index}" for index in range(query_count)],
                source_priorities=["Government statistics", "Company filings"],
                uses_uploads=uses_uploads,
            )
        ],
        source_strategy=SourceStrategy(
            preferred_source_types=["Government statistics", "Company filings"],
            freshness_requirements="Prefer the most recent complete calendar year.",
            corroboration_rules=["Corroborate material estimates with an independent source."],
        ),
        outline=[
            ReportSection(
                id="market_findings",
                title="Market findings",
                purpose="Present market size and direction.",
                research_question_ids=["market_size"],
            )
        ],
        diagram_candidates=[
            DiagramCandidate(
                section_id="market_findings",
                diagram_type="timeline",
                purpose="Show changes in market size over time.",
            )
        ],
    )


def make_plan(*, preset: DepthPreset = DepthPreset.STANDARD):
    from deep_research.contracts.planning import ResearchPlan

    return ResearchPlan.finalize(
        draft=make_plan_draft(),
        brief=ResearchBrief(topic="EV market"),
        budget=BudgetLimits.for_preset(preset),
        version=1,
    )


def make_evidence_package() -> EvidencePackage:
    return EvidencePackage(
        sources=[
            SourceRecord(
                source_id="S1",
                source_type=SourceType.WEB_PAGE,
                title="Official EV statistics",
                publisher="National Statistics Office",
                publication_date="2025",
                access_date="2026-08-26",
                canonical_url="https://example.gov/ev-statistics",
                content_hash="a" * 64,
            )
        ],
        excerpts=[
            EvidenceExcerpt(
                evidence_id="E1",
                source_id="S1",
                excerpt="EV sales increased by 20 percent in 2025.",
                location="Table 2",
            )
        ],
        claims=[
            EvidenceClaim(
                claim_id="C1",
                research_question_id="market_size",
                section_ids=["market_findings"],
                normalized_claim="EV sales increased by 20 percent in 2025.",
                evidence_ids=["E1"],
                support_strength=SupportStrength.STRONG,
            )
        ],
    )


def make_review_draft(
    *,
    coverage: CoverageStatus = CoverageStatus.COVERED,
    recommends_approval: bool = True,
    with_retry: bool = False,
) -> ReviewDraft:
    payload = {
        "question_coverage": [
            CoverageAssessment(
                item_id="market_size",
                status=coverage,
                rationale="The accepted evidence addresses the market-size question.",
            )
        ],
        "section_coverage": [
            CoverageAssessment(
                item_id="market_findings",
                status=coverage,
                rationale="The evidence supports the planned market section.",
            )
        ],
        "source_scores": [
            SourceQualityScore(
                source_id="S1",
                authority=0.9,
                freshness=0.9,
                relevance=1,
                independence=0.8,
                accessibility=1,
                rationale="This is accessible primary statistical evidence.",
            )
        ],
        "recommends_approval": recommends_approval,
    }
    if with_retry:
        payload["retry_tasks"] = [
            {
                "task_id": "repair_market_size",
                "objective": "Find a second primary market-size estimate.",
                "research_question_ids": ["market_size"],
                "section_ids": ["market_findings"],
                "candidate_queries": ["official EV sales 2025"],
            }
        ]
    return ReviewDraft.model_validate(payload)


def make_report_draft(
    *, content: str = "EV sales increased by 20 percent in 2025. [S1]"
) -> ReportDraft:
    return ReportDraft(
        title="EV Market Report",
        executive_summary="EV adoption increased according to the accepted evidence. [S1]",
        methodology="The report synthesizes the supplied normalized evidence.",
        sections=[
            ReportSectionDraft(
                section_id="market_findings",
                title="Market findings",
                content=content,
            )
        ],
        limitations=[],
        conclusion="The measured market expanded during the observed period. [S1]",
        follow_up_topics=["Compare growth across regions."],
    )


def make_report_context() -> ReportContext:
    return ReportContext(
        title="EV market report",
        executive_summary="The market is growing, with meaningful regional variation.",
        sections=[
            ReportSectionSummary(
                section_id="market_findings",
                title="Market findings",
                summary="Adoption and growth differ by geography.",
            )
        ],
        conclusion="Infrastructure and policy are major variables.",
        limitations=["Comparable regional data is incomplete."],
    )


def make_question_set(
    *, count: int = 5, originating_section_id: str = "market_findings"
) -> FollowUpQuestionSet:
    return FollowUpQuestionSet(
        questions=[
            FollowUpQuestion(
                question=f"What additional market factor should be evaluated in study {index}?",
                rationale="It may change the report's decision implications.",
                originating_section_id=originating_section_id,
                priority=index,
            )
            for index in range(1, count + 1)
        ]
    )
