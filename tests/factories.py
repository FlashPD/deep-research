from deep_research.contracts.planning import (
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
