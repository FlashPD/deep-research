from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmpty = Annotated[str, Field(min_length=1, max_length=4_000)]


class AnswerType(StrEnum):
    TEXT = "text"
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    DATE_RANGE = "date_range"
    CONFIRMATION = "confirmation"


class UploadMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: NonEmpty
    filename: NonEmpty
    media_type: NonEmpty
    size_bytes: int = Field(ge=0)


class ResearchBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: NonEmpty
    objective: str | None = Field(default=None, max_length=4_000)
    audience: str | None = Field(default=None, max_length=1_000)
    scope: list[str] = Field(default_factory=list, max_length=30)
    exclusions: list[str] = Field(default_factory=list, max_length=30)
    time_range: str | None = Field(default=None, max_length=500)
    geography: list[str] = Field(default_factory=list, max_length=30)
    definitions: dict[str, str] = Field(default_factory=dict)
    comparison_criteria: list[str] = Field(default_factory=list, max_length=30)
    desired_decision: str | None = Field(default=None, max_length=2_000)
    output_expectations: list[str] = Field(default_factory=list, max_length=30)
    assumptions: list[str] = Field(default_factory=list, max_length=30)


class ClarificationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    question: NonEmpty
    rationale: Annotated[str, Field(min_length=1, max_length=1_000)]
    expected_answer_type: AnswerType
    options: list[str] = Field(default_factory=list, max_length=12)
    required: bool = True

    @model_validator(mode="after")
    def validate_options(self) -> "ClarificationQuestion":
        select_types = {AnswerType.SINGLE_SELECT, AnswerType.MULTI_SELECT}
        if self.expected_answer_type in select_types and len(self.options) < 2:
            raise ValueError("select questions require at least two options")
        if self.expected_answer_type not in select_types and self.options:
            raise ValueError("options are only valid for select questions")
        return self


class ClarificationAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: NonEmpty
    value: str | list[str] | bool


class ClarifierRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: NonEmpty
    uploads: list[UploadMetadata] = Field(default_factory=list, max_length=10)
    answers: list[ClarificationAnswer] = Field(default_factory=list, max_length=15)
    current_brief: ResearchBrief | None = None
    round_number: int = Field(default=1, ge=1, le=3)


class ClarificationDecision(BaseModel):
    """The only structured output accepted from the clarifier model."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["needs_clarification", "scope_ready"]
    brief: ResearchBrief
    questions: list[ClarificationQuestion] = Field(default_factory=list, max_length=5)
    interpretation_summary: Annotated[str, Field(min_length=1, max_length=2_000)]

    @model_validator(mode="after")
    def validate_status_shape(self) -> "ClarificationDecision":
        if self.status == "needs_clarification" and not self.questions:
            raise ValueError("needs_clarification requires at least one question")
        if self.status == "scope_ready" and self.questions:
            raise ValueError("scope_ready cannot contain questions")
        return self
