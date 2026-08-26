from typing import Annotated

from fastapi import APIRouter, Depends, Request

from deep_research.agents.clarifier import ClarifierAgent
from deep_research.contracts.clarification import ClarificationDecision, ClarifierRequest

router = APIRouter(prefix="/v1")


def _get_clarifier(request: Request) -> ClarifierAgent:
    return request.app.state.clarifier


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/clarifier/evaluate", response_model=ClarificationDecision)
async def evaluate_clarification(
    payload: ClarifierRequest,
    clarifier: Annotated[ClarifierAgent, Depends(_get_clarifier)],
) -> ClarificationDecision:
    return await clarifier.evaluate(payload)
