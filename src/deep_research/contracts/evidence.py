import hashlib
import json
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deep_research.contracts.planning import ResearchPlan

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
SourceId = Annotated[str, Field(pattern=r"^S[1-9][0-9]*$")]
EvidenceId = Annotated[str, Field(pattern=r"^E[1-9][0-9]*$")]
ClaimId = Annotated[str, Field(pattern=r"^C[1-9][0-9]*$")]
ShortText = Annotated[str, Field(min_length=1, max_length=4_000)]
ReviewText = Annotated[str, Field(min_length=1, max_length=500)]


class SourceType(StrEnum):
    WEB_PAGE = "web_page"
    PUBLIC_PDF = "public_pdf"
    UPLOAD = "upload"


class SupportStrength(StrEnum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class ReviewState(StrEnum):
    APPROVED = "approved"
    REPAIR_REQUIRED = "repair_required"
    APPROVED_WITH_LIMITATIONS = "approved_with_limitations"
    REJECTED = "rejected"


class CoverageStatus(StrEnum):
    COVERED = "covered"
    PARTIAL = "partial"
    MISSING = "missing"


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: SourceId
    source_type: SourceType
    title: ShortText
    publisher: str | None = Field(default=None, max_length=500)
    author: str | None = Field(default=None, max_length=500)
    publication_date: str | None = Field(default=None, max_length=100)
    access_date: Annotated[str, Field(min_length=1, max_length=100)]
    canonical_url: str | None = Field(default=None, pattern=r"^https?://", max_length=4_000)
    upload_name: str | None = Field(default=None, max_length=500)
    content_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    quality_flags: list[ShortText] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_locator(self) -> Self:
        if self.source_type is SourceType.UPLOAD:
            if not self.upload_name or self.canonical_url:
                raise ValueError("upload sources require upload_name and cannot contain a URL")
        elif not self.canonical_url or self.upload_name:
            raise ValueError("public sources require canonical_url and cannot contain upload_name")
        return self


class EvidenceExcerpt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: EvidenceId
    source_id: SourceId
    excerpt: Annotated[str, Field(min_length=1, max_length=4_000)]
    location: ShortText


class EvidenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: ClaimId
    research_question_id: Identifier
    section_ids: list[Identifier] = Field(min_length=1, max_length=20)
    normalized_claim: ShortText
    evidence_ids: list[EvidenceId] = Field(default_factory=list, max_length=30)
    support_strength: SupportStrength
    contradictions: list[ShortText] = Field(default_factory=list, max_length=20)
    is_inference: bool = False

    @model_validator(mode="after")
    def validate_support(self) -> Self:
        if not self.evidence_ids and not self.is_inference:
            raise ValueError("a non-inference claim requires at least one evidence excerpt")
        return self


class EvidencePackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SourceRecord] = Field(default_factory=list, max_length=100)
    excerpts: list[EvidenceExcerpt] = Field(default_factory=list, max_length=1_000)
    claims: list[EvidenceClaim] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        source_ids = _unique(self.sources, "source_id", "source")
        evidence_ids = _unique(self.excerpts, "evidence_id", "evidence excerpt")
        _unique(self.claims, "claim_id", "claim")
        unknown_excerpt_sources = {item.source_id for item in self.excerpts} - source_ids
        if unknown_excerpt_sources:
            raise ValueError(
                f"evidence excerpts reference unknown sources: {sorted(unknown_excerpt_sources)}"
            )
        unknown_claim_evidence = {
            evidence_id
            for claim in self.claims
            for evidence_id in claim.evidence_ids
            if evidence_id not in evidence_ids
        }
        if unknown_claim_evidence:
            raise ValueError(
                f"claims reference unknown evidence excerpts: {sorted(unknown_claim_evidence)}"
            )
        return self

    def source_ids_for_claim(self, claim: EvidenceClaim) -> set[str]:
        excerpt_sources = {item.evidence_id: item.source_id for item in self.excerpts}
        return {excerpt_sources[evidence_id] for evidence_id in claim.evidence_ids}

    def calculate_checksum(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


class BudgetUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elapsed_seconds: float = Field(default=0, ge=0)
    searches: int = Field(default=0, ge=0)
    fetched_sources: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)


class CoverageAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: Identifier
    status: CoverageStatus
    rationale: ReviewText


class SourceQualityScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: SourceId
    authority: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    independence: float = Field(ge=0, le=1)
    accessibility: float = Field(ge=0, le=1)
    rationale: ReviewText


class UnsupportedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: ClaimId
    rationale: ReviewText
    material: bool = True


class EvidenceContradiction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_ids: list[ClaimId] = Field(min_length=1, max_length=20)
    description: ReviewText
    resolution: str | None = Field(default=None, max_length=500)


class EvidenceRepairTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Identifier
    objective: ReviewText
    research_question_ids: list[Identifier] = Field(min_length=1, max_length=20)
    section_ids: list[Identifier] = Field(min_length=1, max_length=20)
    candidate_queries: list[ShortText] = Field(default_factory=list, max_length=10)


class ReviewDraft(BaseModel):
    """Model-authored semantic review; retry bounds and final state are application-owned."""

    model_config = ConfigDict(extra="forbid")

    question_coverage: list[CoverageAssessment]
    section_coverage: list[CoverageAssessment]
    source_scores: list[SourceQualityScore]
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list, max_length=200)
    contradictions: list[EvidenceContradiction] = Field(default_factory=list, max_length=100)
    overconfident_claim_ids: list[ClaimId] = Field(default_factory=list, max_length=200)
    retry_tasks: list[EvidenceRepairTask] = Field(default_factory=list, max_length=20)
    limitations: list[ReviewText] = Field(default_factory=list, max_length=50)
    recommends_approval: bool


class ReviewResult(ReviewDraft):
    review_state: ReviewState
    repair_round: int = Field(ge=0)
    coverage_score: float = Field(ge=0, le=1)
    citation_coverage: float = Field(ge=0, le=1)
    deterministic_issues: list[ShortText] = Field(default_factory=list, max_length=200)
    plan_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    evidence_checksum: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class ReviewerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: ResearchPlan
    evidence: EvidencePackage
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)
    repair_round: int = Field(default=0, ge=0)


def _unique(items: list[BaseModel], field: str, label: str) -> set[str]:
    values = [str(getattr(item, field)) for item in items]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} IDs must be unique")
    return set(values)
