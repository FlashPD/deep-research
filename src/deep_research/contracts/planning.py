import hashlib
import json
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deep_research.contracts.clarification import ResearchBrief, UploadMetadata

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
ShortText = Annotated[str, Field(min_length=1, max_length=2_000)]


class DepthPreset(StrEnum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class BudgetLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    preset: DepthPreset
    target_duration_seconds: int = Field(gt=0)
    max_search_queries: int = Field(gt=0)
    max_accepted_sources: int = Field(gt=0)
    research_concurrency: int = Field(gt=0)
    reviewer_retries: int = Field(ge=0)
    adaptive_search_queries: int = Field(default=0, ge=0, le=2)

    @property
    def absolute_search_query_ceiling(self) -> int:
        return self.max_search_queries + self.adaptive_search_queries

    @classmethod
    def for_preset(cls, preset: DepthPreset) -> Self:
        values = {
            DepthPreset.QUICK: (300, 5, 10, 3, 1, 1),
            DepthPreset.STANDARD: (900, 20, 30, 6, 2, 0),
            DepthPreset.DEEP: (2_700, 50, 75, 10, 2, 0),
        }
        duration, searches, sources, concurrency, retries, adaptive_searches = values[preset]
        return cls(
            preset=preset,
            target_duration_seconds=duration,
            max_search_queries=searches,
            max_accepted_sources=sources,
            research_concurrency=concurrency,
            reviewer_retries=retries,
            adaptive_search_queries=adaptive_searches,
        )


class ResearchQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Identifier
    question: ShortText
    success_criteria: list[ShortText] = Field(min_length=1, max_length=10)


class Workstream(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Identifier
    title: ShortText
    objective: ShortText
    research_question_ids: list[Identifier] = Field(min_length=1, max_length=20)
    dependencies: list[Identifier] = Field(default_factory=list, max_length=20)
    candidate_queries: list[ShortText] = Field(default_factory=list, max_length=20)
    source_priorities: list[ShortText] = Field(min_length=1, max_length=10)
    uses_uploads: bool = False


class SourceStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_source_types: list[ShortText] = Field(min_length=1, max_length=12)
    freshness_requirements: ShortText
    corroboration_rules: list[ShortText] = Field(min_length=1, max_length=10)


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Identifier
    title: ShortText
    purpose: ShortText
    research_question_ids: list[Identifier] = Field(default_factory=list, max_length=20)


class DiagramCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: Identifier
    diagram_type: Annotated[str, Field(min_length=1, max_length=100)]
    purpose: ShortText


class ResearchPlanDraft(BaseModel):
    """Model-authored fields. Control-plane fields are added deterministically."""

    model_config = ConfigDict(extra="forbid")

    questions: list[ResearchQuestion] = Field(min_length=1, max_length=20)
    workstreams: list[Workstream] = Field(min_length=1, max_length=20)
    source_strategy: SourceStrategy
    outline: list[ReportSection] = Field(min_length=1, max_length=20)
    diagram_candidates: list[DiagramCandidate] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_graph_and_references(self) -> Self:
        question_ids = _unique_ids(self.questions, "research question")
        workstream_ids = _unique_ids(self.workstreams, "workstream")
        section_ids = _unique_ids(self.outline, "report section")

        assigned_questions: set[str] = set()
        dependencies: dict[str, list[str]] = {}
        for workstream in self.workstreams:
            unknown_questions = set(workstream.research_question_ids) - question_ids
            if unknown_questions:
                raise ValueError(
                    f"workstream {workstream.id!r} references unknown questions: "
                    f"{sorted(unknown_questions)}"
                )
            unknown_dependencies = set(workstream.dependencies) - workstream_ids
            if unknown_dependencies:
                raise ValueError(
                    f"workstream {workstream.id!r} references unknown dependencies: "
                    f"{sorted(unknown_dependencies)}"
                )
            if workstream.id in workstream.dependencies:
                raise ValueError(f"workstream {workstream.id!r} cannot depend on itself")
            assigned_questions.update(workstream.research_question_ids)
            dependencies[workstream.id] = workstream.dependencies

        unassigned = question_ids - assigned_questions
        if unassigned:
            raise ValueError(
                f"research questions are not assigned to workstreams: {sorted(unassigned)}"
            )
        _validate_acyclic(dependencies)

        outlined_questions = {
            question_id for section in self.outline for question_id in section.research_question_ids
        }
        unknown_outline_questions = outlined_questions - question_ids
        if unknown_outline_questions:
            raise ValueError(
                f"report outline references unknown questions: {sorted(unknown_outline_questions)}"
            )
        unoutlined_questions = question_ids - outlined_questions
        if unoutlined_questions:
            raise ValueError(
                f"research questions are not represented in the report outline: "
                f"{sorted(unoutlined_questions)}"
            )
        unknown_diagram_sections = {
            candidate.section_id for candidate in self.diagram_candidates
        } - section_ids
        if unknown_diagram_sections:
            raise ValueError(
                f"diagram candidates reference unknown sections: {sorted(unknown_diagram_sections)}"
            )
        return self


class ResearchPlan(ResearchPlanDraft):
    version: int = Field(ge=1)
    brief: ResearchBrief
    budget: BudgetLimits
    content_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]

    @model_validator(mode="after")
    def validate_content_hash(self) -> Self:
        if self.content_hash != self.calculate_hash():
            raise ValueError("content_hash does not match the canonical plan content")
        return self

    def calculate_hash(self) -> str:
        content = self.model_dump(mode="json", exclude={"content_hash"})
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def finalize(
        cls,
        *,
        draft: ResearchPlanDraft,
        brief: ResearchBrief,
        budget: BudgetLimits,
        version: int,
    ) -> Self:
        content = {
            **draft.model_dump(mode="python"),
            "version": version,
            "brief": brief,
            "budget": budget,
        }
        json_content = {
            **draft.model_dump(mode="json"),
            "version": version,
            "brief": brief.model_dump(mode="json"),
            "budget": budget.model_dump(mode="json"),
        }
        canonical = json.dumps(
            json_content,
            sort_keys=True,
            separators=(",", ":"),
        )
        content_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return cls.model_validate({**content, "content_hash": content_hash})


class PlannerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief: ResearchBrief
    depth: DepthPreset = DepthPreset.STANDARD
    uploads: list[UploadMetadata] = Field(default_factory=list, max_length=10)
    user_edits: list[ShortText] = Field(default_factory=list, max_length=30)
    previous_plan: ResearchPlan | None = None


def _unique_ids(items: list[BaseModel], label: str) -> set[str]:
    ids = [str(item.id) for item in items]  # type: ignore[attr-defined]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} IDs must be unique")
    return set(ids)


def _validate_acyclic(dependencies: dict[str, list[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"workstream dependency cycle detected at {node!r}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in dependencies[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in dependencies:
        visit(node)
