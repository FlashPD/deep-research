from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from deep_research.contracts.clarification import ClarificationDecision
from deep_research.contracts.evidence import EvidencePackage, ReviewResult
from deep_research.contracts.questions import FollowUpQuestionSet
from deep_research.contracts.reporting import ReportArtifact
from deep_research.contracts.research import ResearchResult


class GraphNode(StrEnum):
    CLARIFIER = "clarifier"
    PLANNER = "planner"
    PLAN_APPROVAL = "plan_approval"
    RESEARCH = "research"
    REVIEWER = "reviewer"
    REPORT = "report"
    QUESTIONS = "questions"
    FINALIZE = "finalize"


class GraphCheckpoint(BaseModel):
    """Canonical resumable state for the bounded outer graph."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    completed_nodes: list[GraphNode] = Field(default_factory=list)
    clarification_round: int = Field(default=0, ge=0, le=3)
    clarification: ClarificationDecision | None = None
    plan_version: int | None = Field(default=None, ge=1)
    plan_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    research_results: dict[str, ResearchResult] = Field(default_factory=dict)
    evidence: EvidencePackage | None = None
    review: ReviewResult | None = None
    repair_round: int = Field(default=0, ge=0, le=2)
    report: ReportArtifact | None = None
    questions: FollowUpQuestionSet | None = None
    limitations: list[str] = Field(default_factory=list, max_length=100)
    checkpointed_at: datetime | None = None
