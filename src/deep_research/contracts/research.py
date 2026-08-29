from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deep_research.contracts.evidence import (
    BudgetUsage,
    EvidencePackage,
    EvidenceRepairTask,
    SourceType,
    SupportStrength,
)
from deep_research.contracts.planning import ResearchPlan

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
MaterialId = Annotated[str, Field(pattern=r"^M[1-9][0-9]*$")]
ShortText = Annotated[str, Field(min_length=1, max_length=4_000)]
QueryText = Annotated[str, Field(min_length=1, max_length=1_000)]


class ResearchTool(StrEnum):
    SEARCH_WEB = "search_web"
    FETCH_PAGE = "fetch_page"
    SEARCH_UPLOADS = "search_uploads"


class ResearchTaskStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"


class ResearchTask(BaseModel):
    """Application-authored, approved bounds for one isolated workstream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: Identifier
    plan_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    workstream_id: Identifier
    objective: ShortText
    research_question_ids: list[Identifier] = Field(min_length=1, max_length=20)
    section_ids: list[Identifier] = Field(min_length=1, max_length=20)
    candidate_queries: list[QueryText] = Field(default_factory=list, max_length=50)
    permitted_tools: set[ResearchTool] = Field(min_length=1)
    max_queries: int = Field(ge=0)
    max_sources: int = Field(ge=0)
    max_material_chars: int = Field(default=200_000, ge=10_000, le=1_000_000)
    tool_timeout_seconds: float = Field(default=30, gt=0, le=120)
    attempt: int = Field(default=1, ge=1, le=3)
    status: ResearchTaskStatus = ResearchTaskStatus.PENDING

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        has_search = ResearchTool.SEARCH_WEB in self.permitted_tools
        has_fetch = ResearchTool.FETCH_PAGE in self.permitted_tools
        if has_search != has_fetch:
            raise ValueError("search_web and fetch_page permissions must be granted together")
        if len(self.candidate_queries) > self.max_queries:
            raise ValueError("candidate queries exceed the task query ceiling")
        return self

    @classmethod
    def for_workstream(
        cls,
        plan: ResearchPlan,
        workstream_id: str,
        *,
        max_queries: int | None = None,
        max_sources: int | None = None,
        attempt: int = 1,
    ) -> Self:
        workstream = next(
            (item for item in plan.workstreams if item.id == workstream_id), None
        )
        if workstream is None:
            raise ValueError(f"unknown workstream {workstream_id!r}")
        question_ids = set(workstream.research_question_ids)
        section_ids = [
            section.id
            for section in plan.outline
            if question_ids.intersection(section.research_question_ids)
        ]
        tools = {ResearchTool.SEARCH_WEB, ResearchTool.FETCH_PAGE}
        if workstream.uses_uploads:
            tools.add(ResearchTool.SEARCH_UPLOADS)
        query_limit = (
            len(workstream.candidate_queries) if max_queries is None else max_queries
        )
        source_limit = (
            plan.budget.max_accepted_sources if max_sources is None else max_sources
        )
        if query_limit > plan.budget.max_search_queries:
            raise ValueError("task query ceiling exceeds the approved run ceiling")
        if source_limit > plan.budget.max_accepted_sources:
            raise ValueError("task source ceiling exceeds the approved run ceiling")
        return cls(
            task_id=f"{workstream.id[:52]}_attempt_{attempt}",
            plan_hash=plan.content_hash,
            workstream_id=workstream.id,
            objective=workstream.objective,
            research_question_ids=workstream.research_question_ids,
            section_ids=section_ids,
            candidate_queries=workstream.candidate_queries,
            permitted_tools=tools,
            max_queries=query_limit,
            max_sources=source_limit,
            attempt=attempt,
        )


class WebSearchOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: QueryText
    research_question_ids: list[Identifier] = Field(min_length=1, max_length=20)
    date_range: str | None = Field(default=None, max_length=200)
    domains: list[Annotated[str, Field(min_length=1, max_length=253)]] = Field(
        default_factory=list, max_length=20
    )
    max_results: int = Field(default=5, ge=1, le=10)


class UploadSearchOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: QueryText
    research_question_ids: list[Identifier] = Field(min_length=1, max_length=20)
    max_chunks: int = Field(default=5, ge=1, le=20)


class ResearchExecutionPlan(BaseModel):
    """Model-authored tool plan validated against the application-owned ResearchTask."""

    model_config = ConfigDict(extra="forbid")

    web_searches: list[WebSearchOperation] = Field(default_factory=list, max_length=50)
    upload_searches: list[UploadSearchOperation] = Field(default_factory=list, max_length=50)
    rationale: ShortText


class WebSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: Annotated[str, Field(pattern=r"^https?://", max_length=4_000)]
    title: ShortText
    snippet: str | None = Field(default=None, max_length=4_000)
    published_date: str | None = Field(default=None, max_length=100)


class FetchPageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: Annotated[str, Field(pattern=r"^https?://", max_length=4_000)]


class FetchedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_url: Annotated[str, Field(pattern=r"^https?://", max_length=4_000)]
    title: ShortText
    content: Annotated[str, Field(min_length=1, max_length=100_000)]
    source_type: SourceType = SourceType.WEB_PAGE
    publisher: str | None = Field(default=None, max_length=500)
    author: str | None = Field(default=None, max_length=500)
    publication_date: str | None = Field(default=None, max_length=100)
    access_date: Annotated[str, Field(min_length=1, max_length=100)]
    location: ShortText = "Page text"

    @model_validator(mode="after")
    def validate_public_type(self) -> Self:
        if self.source_type is SourceType.UPLOAD:
            raise ValueError("fetch_page cannot return an upload source")
        return self


class UploadChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: Annotated[str, Field(min_length=1, max_length=500)]
    filename: Annotated[str, Field(min_length=1, max_length=500)]
    title: ShortText
    content: Annotated[str, Field(min_length=1, max_length=50_000)]
    location: ShortText
    document_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    access_date: Annotated[str, Field(min_length=1, max_length=100)]


class CapturedMaterial(BaseModel):
    """Bounded evidence content passed to synthesis; never treated as instructions."""

    model_config = ConfigDict(extra="forbid")

    material_id: MaterialId
    source_type: SourceType
    source_key: ShortText
    title: ShortText
    content: Annotated[str, Field(min_length=1, max_length=100_000)]
    content_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    location: ShortText
    canonical_url: str | None = Field(default=None, pattern=r"^https?://", max_length=4_000)
    upload_name: str | None = Field(default=None, max_length=500)
    publisher: str | None = Field(default=None, max_length=500)
    author: str | None = Field(default=None, max_length=500)
    publication_date: str | None = Field(default=None, max_length=100)
    access_date: Annotated[str, Field(min_length=1, max_length=100)]


class DraftEvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material_id: MaterialId
    excerpt: Annotated[str, Field(min_length=10, max_length=4_000)]
    location: ShortText


class DraftResearchClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research_question_id: Identifier
    section_ids: list[Identifier] = Field(min_length=1, max_length=20)
    normalized_claim: ShortText
    evidence: list[DraftEvidenceSelection] = Field(default_factory=list, max_length=30)
    support_strength: SupportStrength
    contradictions: list[ShortText] = Field(default_factory=list, max_length=20)
    is_inference: bool = False

    @model_validator(mode="after")
    def validate_evidence_requirement(self) -> Self:
        if not self.evidence and not self.is_inference:
            raise ValueError("non-inference research claims require captured evidence")
        return self


class ResearchSynthesisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[DraftResearchClaim] = Field(default_factory=list, max_length=200)
    limitations: list[ShortText] = Field(default_factory=list, max_length=30)


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: ResearchPlan
    task: ResearchTask
    repair_task: EvidenceRepairTask | None = None
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)

    @model_validator(mode="after")
    def validate_task_binding(self) -> Self:
        if self.task.status is not ResearchTaskStatus.PENDING:
            raise ValueError("only pending research tasks can be executed")
        if self.task.plan_hash != self.plan.content_hash:
            raise ValueError("research task belongs to a different plan")
        workstream = next(
            (item for item in self.plan.workstreams if item.id == self.task.workstream_id), None
        )
        if workstream is None:
            raise ValueError("research task references an unknown workstream")
        if set(self.task.research_question_ids) != set(workstream.research_question_ids):
            raise ValueError("research task questions differ from the approved workstream")
        if self.task.objective != workstream.objective:
            raise ValueError("research task objective differs from the approved workstream")
        approved_sections = {
            section.id
            for section in self.plan.outline
            if set(section.research_question_ids).intersection(
                workstream.research_question_ids
            )
        }
        if set(self.task.section_ids) != approved_sections:
            raise ValueError("research task sections differ from the approved workstream")
        if self.task.candidate_queries != workstream.candidate_queries:
            raise ValueError("research task queries differ from the approved workstream")
        expected_task_id = f"{workstream.id[:52]}_attempt_{self.task.attempt}"
        if self.task.task_id != expected_task_id:
            raise ValueError("research task ID is not canonical")
        if self.task.max_queries > self.plan.budget.max_search_queries:
            raise ValueError("research task query ceiling exceeds the plan")
        if self.task.max_sources > self.plan.budget.max_accepted_sources:
            raise ValueError("research task source ceiling exceeds the plan")
        if ResearchTool.SEARCH_UPLOADS in self.task.permitted_tools and not workstream.uses_uploads:
            raise ValueError("research task cannot search uploads for this workstream")
        if self.repair_task is not None:
            if not set(self.repair_task.research_question_ids).issubset(
                self.task.research_question_ids
            ):
                raise ValueError("repair task expands the approved research questions")
            if not set(self.repair_task.section_ids).issubset(self.task.section_ids):
                raise ValueError("repair task expands the approved report sections")
        return self


class ResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Identifier
    status: ResearchTaskStatus
    evidence: EvidencePackage
    budget_usage: BudgetUsage
    limitations: list[ShortText] = Field(default_factory=list, max_length=50)
