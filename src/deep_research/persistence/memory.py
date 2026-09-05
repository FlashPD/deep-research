import asyncio
from datetime import UTC, datetime

from deep_research.contracts.runs import (
    IdempotencyRecord,
    Principal,
    ResearchRun,
    RunEvent,
)
from deep_research.persistence.runs import (
    ConcurrencyConflictError,
    RunNotFoundError,
)


class InMemoryRunRepository:
    def __init__(self) -> None:
        self._runs: dict[tuple[str, str], ResearchRun] = {}
        self._events: dict[tuple[str, str], list[RunEvent]] = {}
        self._idempotency: dict[tuple[str, str, str], IdempotencyRecord] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        run: ResearchRun,
        event: RunEvent,
        idempotency: IdempotencyRecord,
    ) -> None:
        run_key = (run.tenant_id, run.run_id)
        idem_key = (idempotency.tenant_id, idempotency.owner_id, idempotency.key)
        async with self._lock:
            existing_idempotency = self._idempotency.get(idem_key)
            if existing_idempotency and existing_idempotency.expires_at <= datetime.now(UTC):
                del self._idempotency[idem_key]
            if run_key in self._runs or idem_key in self._idempotency:
                raise ConcurrencyConflictError("run or idempotency key already exists")
            self._runs[run_key] = _copy_run(run)
            self._events[run_key] = [_copy_event(event)]
            self._idempotency[idem_key] = idempotency.model_copy(deep=True)

    async def get(self, principal: Principal, run_id: str) -> ResearchRun:
        run = self._runs.get((principal.tenant_id, run_id))
        if run is None or run.owner_id != principal.subject or run.expires_at <= datetime.now(UTC):
            raise RunNotFoundError("run not found")
        return _copy_run(run)

    async def commit(
        self,
        run: ResearchRun,
        event: RunEvent,
        *,
        expected_revision: int,
        idempotency: IdempotencyRecord,
    ) -> None:
        run_key = (run.tenant_id, run.run_id)
        idem_key = (idempotency.tenant_id, idempotency.owner_id, idempotency.key)
        async with self._lock:
            current = self._runs.get(run_key)
            if current is None or current.owner_id != run.owner_id:
                raise RunNotFoundError("run not found")
            if current.revision != expected_revision:
                raise ConcurrencyConflictError("run revision changed")
            existing_idempotency = self._idempotency.get(idem_key)
            if existing_idempotency and existing_idempotency.expires_at <= datetime.now(UTC):
                del self._idempotency[idem_key]
            if idem_key in self._idempotency:
                raise ConcurrencyConflictError("idempotency key already exists")
            if event.cursor != current.next_event_cursor:
                raise ConcurrencyConflictError("event cursor is out of order")
            self._runs[run_key] = _copy_run(run)
            self._events.setdefault(run_key, []).append(_copy_event(event))
            self._idempotency[idem_key] = idempotency.model_copy(deep=True)

    async def get_idempotency(self, principal: Principal, key: str) -> IdempotencyRecord | None:
        record = self._idempotency.get((principal.tenant_id, principal.subject, key))
        if record is None or record.expires_at <= datetime.now(UTC):
            return None
        return record.model_copy(deep=True)

    async def list_events(
        self,
        principal: Principal,
        run_id: str,
        *,
        after_cursor: int = 0,
        limit: int = 100,
    ) -> list[RunEvent]:
        await self.get(principal, run_id)
        return [
            _copy_event(event)
            for event in self._events.get((principal.tenant_id, run_id), [])
            if event.cursor > after_cursor
        ][:limit]


def _copy_run(run: ResearchRun) -> ResearchRun:
    return ResearchRun.model_validate_json(run.model_dump_json())


def _copy_event(event: RunEvent) -> RunEvent:
    return RunEvent.model_validate_json(event.model_dump_json())
