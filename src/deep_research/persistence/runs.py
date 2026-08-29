from typing import Protocol

from deep_research.contracts.runs import (
    IdempotencyRecord,
    Principal,
    ResearchRun,
    RunEvent,
)


class RunNotFoundError(LookupError):
    pass


class ConcurrencyConflictError(RuntimeError):
    pass


class RunRepository(Protocol):
    async def create(
        self,
        run: ResearchRun,
        event: RunEvent,
        idempotency: IdempotencyRecord,
    ) -> None: ...

    async def get(self, principal: Principal, run_id: str) -> ResearchRun: ...

    async def commit(
        self,
        run: ResearchRun,
        event: RunEvent,
        *,
        expected_revision: int,
        idempotency: IdempotencyRecord,
    ) -> None: ...

    async def get_idempotency(self, principal: Principal, key: str) -> IdempotencyRecord | None: ...

    async def list_events(
        self,
        principal: Principal,
        run_id: str,
        *,
        after_cursor: int = 0,
        limit: int = 100,
    ) -> list[RunEvent]: ...
