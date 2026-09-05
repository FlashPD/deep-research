import asyncio
import hashlib
from datetime import UTC, datetime
from typing import Any

from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError

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

_SERIALIZER = TypeSerializer()


class DynamoDBRunRepository:
    """Single-table run repository using atomic run/event/idempotency transactions."""

    def __init__(self, table: Any) -> None:
        self._table = table
        self._client = table.meta.client
        self._table_name = table.name

    async def create(
        self,
        run: ResearchRun,
        event: RunEvent,
        idempotency: IdempotencyRecord,
    ) -> None:
        actions = [
            _put_action(self._table_name, _run_item(run), "attribute_not_exists(PK)"),
            _put_action(
                self._table_name,
                _event_item(run.tenant_id, event, idempotency.expires_at),
                None,
            ),
            _put_action(
                self._table_name,
                _idempotency_item(idempotency),
                "attribute_not_exists(PK) OR expires_at < :now",
                {":now": int(datetime.now(UTC).timestamp())},
            ),
        ]
        await self._transact(actions, idempotency)

    async def get(self, principal: Principal, run_id: str) -> ResearchRun:
        response = await asyncio.to_thread(
            self._table.get_item,
            Key={"PK": _run_pk(principal.tenant_id, run_id), "SK": "META"},
            ConsistentRead=True,
        )
        item = response.get("Item")
        if item is None or item.get("owner_id") != principal.subject:
            raise RunNotFoundError("run not found")
        run = ResearchRun.model_validate_json(item["document"])
        if run.expires_at <= datetime.now(UTC):
            raise RunNotFoundError("run not found")
        return run

    async def commit(
        self,
        run: ResearchRun,
        event: RunEvent,
        *,
        expected_revision: int,
        idempotency: IdempotencyRecord,
    ) -> None:
        actions = [
            _put_action(
                self._table_name,
                _run_item(run),
                "revision = :revision AND owner_id = :owner",
                {":revision": expected_revision, ":owner": run.owner_id},
            ),
            _put_action(
                self._table_name,
                _event_item(run.tenant_id, event, idempotency.expires_at),
                "attribute_not_exists(PK)",
            ),
            _put_action(
                self._table_name,
                _idempotency_item(idempotency),
                "attribute_not_exists(PK) OR expires_at < :now",
                {":now": int(datetime.now(UTC).timestamp())},
            ),
        ]
        await self._transact(actions, idempotency)

    async def get_idempotency(self, principal: Principal, key: str) -> IdempotencyRecord | None:
        response = await asyncio.to_thread(
            self._table.get_item,
            Key={
                "PK": _owner_pk(principal.tenant_id, principal.subject),
                "SK": f"IDEMPOTENCY#{key}",
            },
            ConsistentRead=True,
        )
        item = response.get("Item")
        if item is None:
            return None
        record = IdempotencyRecord.model_validate_json(item["document"])
        if record.expires_at <= datetime.now(UTC):
            return None
        return record

    async def list_events(
        self,
        principal: Principal,
        run_id: str,
        *,
        after_cursor: int = 0,
        limit: int = 100,
    ) -> list[RunEvent]:
        await self.get(principal, run_id)
        response = await asyncio.to_thread(
            self._table.query,
            KeyConditionExpression="PK = :pk AND SK BETWEEN :start AND :end",
            ExpressionAttributeValues={
                ":pk": _run_pk(principal.tenant_id, run_id),
                ":start": _event_sk(after_cursor + 1),
                ":end": "EVENT#99999999999999999999",
            },
            Limit=limit,
            ScanIndexForward=True,
            ConsistentRead=True,
        )
        return [RunEvent.model_validate_json(item["document"]) for item in response["Items"]]

    async def _transact(
        self, actions: list[dict[str, Any]], idempotency: IdempotencyRecord
    ) -> None:
        token = hashlib.sha256(
            f"{idempotency.tenant_id}:{idempotency.owner_id}:{idempotency.key}".encode()
        ).hexdigest()[:36]
        try:
            await asyncio.to_thread(
                self._client.transact_write_items,
                TransactItems=actions,
                ClientRequestToken=token,
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "ConditionalCheckFailedException",
                "TransactionCanceledException",
                "IdempotentParameterMismatchException",
            }:
                raise ConcurrencyConflictError("atomic run commit conflicted") from exc
            raise


def _run_pk(tenant_id: str, run_id: str) -> str:
    return f"TENANT#{tenant_id}#RUN#{run_id}"


def _owner_pk(tenant_id: str, owner_id: str) -> str:
    return f"TENANT#{tenant_id}#OWNER#{owner_id}"


def _event_sk(cursor: int) -> str:
    return f"EVENT#{cursor:020d}"


def _run_item(run: ResearchRun) -> dict[str, Any]:
    return {
        "PK": _run_pk(run.tenant_id, run.run_id),
        "SK": "META",
        "entity_type": "run",
        "owner_id": run.owner_id,
        "revision": run.revision,
        "expires_at": int(run.expires_at.timestamp()),
        "document": run.model_dump_json(),
    }


def _event_item(tenant_id: str, event: RunEvent, expires_at: datetime) -> dict[str, Any]:
    return {
        "PK": _run_pk(tenant_id, event.run_id),
        "SK": _event_sk(event.cursor),
        "entity_type": "event",
        "expires_at": int(expires_at.timestamp()),
        "document": event.model_dump_json(),
    }


def _idempotency_item(record: IdempotencyRecord) -> dict[str, Any]:
    return {
        "PK": _owner_pk(record.tenant_id, record.owner_id),
        "SK": f"IDEMPOTENCY#{record.key}",
        "entity_type": "idempotency",
        "expires_at": int(record.expires_at.timestamp()),
        "document": record.model_dump_json(),
    }


def _put_action(
    table_name: str,
    item: dict[str, Any],
    condition: str | None,
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    put: dict[str, Any] = {
        "TableName": table_name,
        "Item": {key: _SERIALIZER.serialize(value) for key, value in item.items()},
    }
    if condition:
        put["ConditionExpression"] = condition
    if values:
        put["ExpressionAttributeValues"] = {
            key: _SERIALIZER.serialize(value) for key, value in values.items()
        }
    return {"Put": put}
