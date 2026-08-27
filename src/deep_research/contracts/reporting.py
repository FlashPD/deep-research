from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deep_research.contracts.evidence import (
    BudgetUsage,
    ClaimId,
    EvidencePackage,
    ReviewResult,
    SourceId,
)
from deep_research.contracts.planning import ResearchPlan

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
MarkdownText = Annotated[str, Field(min_length=1, max_length=100_000)]
ShortText = Annotated[str, Field(min_length=1, max_length=4_000)]


class MermaidDiagram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: Identifier
    title: Annotated[str, Field(min_length=1, max_length=300)]
    code: Annotated[str, Field(min_length=1, max_length=20_000)]
    fallback_text: ShortText


class ReportSectionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: Identifier
    title: Annotated[str, Field(min_length=1, max_length=300)]
    content: MarkdownText


class ReportedContradiction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_ids: list[ClaimId] = Field(min_length=1, max_length=20)
    summary: ShortText


class ReportDraft(BaseModel):
    """Structured content authored by the model before deterministic Markdown assembly."""

    model_config = ConfigDict(extra="forbid")

    title: Annotated[str, Field(min_length=1, max_length=500)]
    executive_summary: MarkdownText
    methodology: MarkdownText
    sections: list[ReportSectionDraft] = Field(min_length=1, max_length=30)
    diagrams: list[MermaidDiagram] = Field(default_factory=list, max_length=8)
    limitations: list[ShortText] = Field(default_factory=list, max_length=50)
    contradictions: list[ReportedContradiction] = Field(default_factory=list, max_length=100)
    conclusion: MarkdownText
    follow_up_topics: list[ShortText] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_unique_sections(self) -> Self:
        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("report section IDs must be unique")
        return self


class MermaidValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: Identifier
    valid: bool
    error: str | None = Field(default=None, max_length=2_000)


class ReportGenerationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_version: Annotated[str, Field(min_length=1, max_length=100)]
    plan_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    review_round: int = Field(ge=0)


class ReportArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    markdown: MarkdownText
    checksum: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    cited_source_ids: list[SourceId]
    mermaid_validation: list[MermaidValidation]
    generation_metadata: ReportGenerationMetadata


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: ResearchPlan
    evidence: EvidencePackage
    review: ReviewResult
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)

    @model_validator(mode="after")
    def validate_review_binding(self) -> Self:
        if self.review.plan_hash != self.plan.content_hash:
            raise ValueError("review result belongs to a different research plan")
        if self.review.evidence_checksum != self.evidence.calculate_checksum():
            raise ValueError("review result belongs to a different evidence package")
        return self
