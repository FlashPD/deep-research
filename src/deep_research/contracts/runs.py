from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deep_research.contracts.clarification import ClarificationAnswer, ResearchBrief
from deep_research.contracts.evidence import BudgetUsage
from deep_research.contracts.orchestration import GraphCheckpoint
from deep_research.contracts.planning import DepthPreset, ResearchPlan

RunId = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
ShortText = Annotated[str, Field(min_length=1, max_length=4_000)]


class RunState(StrEnum):
    DRAFT = "DRAFT"
    CLARIFYING = "CLARIFYING"
    PLANNING = "PLANNING"
    AWAITING_PLAN_APPROVAL = "AWAITING_PLAN_APPROVAL"
    RESEARCHING = "RESEARCHING"
    REVIEWING = "REVIEWING"
    GENERATING_REPORT = "GENERATING_REPORT"
    GENERATING_QUESTIONS = "GENERATING_QUESTIONS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

    @property
    def terminal(self) -> bool:
        return self in {
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.CANCELLED,
            RunState.EXPIRED,
        }


class Principal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: Annotated[str, Field(min_length=1, max_length=500)]
    tenant_id: Annotated[str, Field(min_length=1, max_length=500)]
    scopes: frozenset[str] = Field(default_factory=frozenset)


class ReportPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audience: str | None = Field(default=None, max_length=1_000)
    tone: str | None = Field(default=None, max_length=200)
    include_diagrams: bool = True


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: ShortText
    depth: DepthPreset = DepthPreset.STANDARD
    report_preferences: ReportPreferences = Field(default_factory=ReportPreferences)


class ResearchRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: RunId
    owner_id: Annotated[str, Field(min_length=1, max_length=500)]
    tenant_id: Annotated[str, Field(min_length=1, max_length=500)]
    topic: ShortText
    depth: DepthPreset
    report_preferences: ReportPreferences
    state: RunState
    revision: int = Field(ge=1)
    next_event_cursor: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    brief: ResearchBrief | None = None
    plan: ResearchPlan | None = None
    approved_plan_version: int | None = Field(default=None, ge=1)
    approved_plan_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)
    graph_checkpoint: GraphCheckpoint = Field(default_factory=GraphCheckpoint)
    failure_code: str | None = Field(default=None, max_length=200)
    failure_message: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def validate_approval(self) -> Self:
        has_version = self.approved_plan_version is not None
        has_hash = self.approved_plan_hash is not None
        if has_version != has_hash:
            raise ValueError("approved plan version and hash must be set together")
        if has_version and self.plan is None:
            raise ValueError("an approved plan must be present on the run")
        return self


class PlanApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    content_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class ClarificationAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_number: int = Field(ge=1, le=3)
    answers: list[ClarificationAnswer] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_unique_questions(self) -> Self:
        question_ids = [answer.question_id for answer in self.answers]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("each clarification question may be answered only once")
        return self


class FailureUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    message: Annotated[str, Field(min_length=1, max_length=2_000)]


class RunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: RunId
    cursor: int = Field(ge=1)
    event_type: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,99}$")]
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = Field(default=None, max_length=200)


class RunEventPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[RunEvent]
    next_cursor: int = Field(ge=0)


class IdempotencyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    owner_id: str
    key: Annotated[str, Field(min_length=8, max_length=200)]
    fingerprint: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    response_json: str
    expires_at: datetime
