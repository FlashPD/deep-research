from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ShortText = Annotated[str, Field(min_length=1, max_length=2_000)]


class ReportSectionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    title: Annotated[str, Field(min_length=1, max_length=300)]
    summary: ShortText


class ReportContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Annotated[str, Field(min_length=1, max_length=500)]
    executive_summary: ShortText
    sections: list[ReportSectionSummary] = Field(min_length=1, max_length=30)
    conclusion: ShortText
    limitations: list[ShortText] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_unique_sections(self) -> Self:
        ids = [section.section_id for section in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError("report section IDs must be unique")
        return self


class FollowUpQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, Field(min_length=1, max_length=1_000)]
    rationale: ShortText
    originating_section_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    priority: int = Field(ge=1, le=10)


class FollowUpQuestionSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[FollowUpQuestion] = Field(min_length=5, max_length=10)

    @model_validator(mode="after")
    def validate_unique_questions_and_priorities(self) -> Self:
        normalized = [item.question.casefold().strip() for item in self.questions]
        priorities = [item.priority for item in self.questions]
        if len(normalized) != len(set(normalized)):
            raise ValueError("follow-up questions must be unique")
        if len(priorities) != len(set(priorities)):
            raise ValueError("follow-up priorities must be unique")
        return self


class QuestionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report: ReportContext
    count: int = Field(default=7, ge=5, le=10)
