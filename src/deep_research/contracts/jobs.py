from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JobPhase(StrEnum):
    CLARIFY = "clarify"
    PLAN = "plan"
    RESEARCH = "research"
    REVIEW = "review"
    REPORT = "report"
    QUESTIONS = "questions"
    FINALIZE = "finalize"


class PhaseJob(BaseModel):
    """Versioned, tenant-bound queue message. Queue metadata is not part of the payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    job_id: UUID = Field(default_factory=uuid4)
    phase: JobPhase
    run_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    tenant_id: Annotated[str, Field(min_length=1, max_length=500)]
    owner_id: Annotated[str, Field(min_length=1, max_length=500)]
    expected_revision: int = Field(ge=1)
    plan_version: int | None = Field(default=None, ge=1)
    plan_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    trace_id: str | None = Field(default=None, max_length=200)
    enqueued_at: datetime

    @model_validator(mode="after")
    def validate_plan_binding(self) -> Self:
        has_version = self.plan_version is not None
        has_hash = self.plan_hash is not None
        if has_version != has_hash:
            raise ValueError("job plan version and hash must be set together")
        if (
            self.phase
            in {
                JobPhase.RESEARCH,
                JobPhase.REVIEW,
                JobPhase.REPORT,
                JobPhase.QUESTIONS,
                JobPhase.FINALIZE,
            }
            and not has_version
        ):
            raise ValueError("post-approval jobs require an exact plan binding")
        return self

    def idempotency_key(self, operation: str) -> str:
        return f"worker:{self.job_id}:{operation}"


class JobDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    job: PhaseJob
    receipt_handle: str
    delivery_count: int = Field(ge=1)
