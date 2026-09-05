from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from boto3.dynamodb.types import TypeDeserializer

from deep_research.contracts.planning import DepthPreset
from deep_research.contracts.runs import (
    IdempotencyRecord,
    ReportPreferences,
    ResearchRun,
    RunEvent,
    RunState,
)
from deep_research.persistence.dynamodb import DynamoDBRunRepository


class FakeDynamoClient:
    def __init__(self) -> None:
        self.transactions: list[dict] = []

    def transact_write_items(self, **kwargs) -> None:
        self.transactions.append(kwargs)


class FakeDynamoTable:
    def __init__(self) -> None:
        self.name = "research-runs"
        self.client = FakeDynamoClient()
        self.meta = SimpleNamespace(client=self.client)


def _records() -> tuple[ResearchRun, RunEvent, IdempotencyRecord]:
    now = datetime.now(UTC)
    expires_at = now + timedelta(days=30)
    run = ResearchRun(
        run_id="a" * 32,
        owner_id="user-1",
        tenant_id="tenant-1",
        topic="Grid storage",
        depth=DepthPreset.STANDARD,
        report_preferences=ReportPreferences(),
        state=RunState.DRAFT,
        revision=1,
        next_event_cursor=2,
        created_at=now,
        updated_at=now,
        expires_at=expires_at,
    )
    event = RunEvent(
        run_id=run.run_id,
        cursor=1,
        event_type="run.created",
        timestamp=now,
    )
    idempotency = IdempotencyRecord(
        tenant_id=run.tenant_id,
        owner_id=run.owner_id,
        key="create-key",
        fingerprint="f" * 64,
        response_json=run.model_dump_json(),
        expires_at=expires_at,
    )
    return run, event, idempotency


@pytest.mark.asyncio
async def test_create_is_an_atomic_run_event_and_idempotency_transaction() -> None:
    table = FakeDynamoTable()
    repository = DynamoDBRunRepository(table)
    run, event, idempotency = _records()

    await repository.create(run, event, idempotency)

    transaction = table.client.transactions[0]
    assert len(transaction["TransactItems"]) == 3
    assert len(transaction["ClientRequestToken"]) == 36
    deserialize = TypeDeserializer().deserialize
    items = [
        {key: deserialize(value) for key, value in action["Put"]["Item"].items()}
        for action in transaction["TransactItems"]
    ]
    run_item, event_item, idempotency_item = items
    assert run_item["PK"] == f"TENANT#tenant-1#RUN#{run.run_id}"
    assert event_item["SK"] == "EVENT#00000000000000000001"
    assert event_item["expires_at"] == int(run.expires_at.timestamp())
    assert idempotency_item["PK"] == "TENANT#tenant-1#OWNER#user-1"
    assert all(
        action["Put"].get("ConditionExpression")
        for action in transaction["TransactItems"]
        if action["Put"]["Item"]["SK"] != {"S": "EVENT#00000000000000000001"}
    )


@pytest.mark.asyncio
async def test_commit_uses_revision_and_owner_compare_and_swap() -> None:
    table = FakeDynamoTable()
    repository = DynamoDBRunRepository(table)
    run, event, idempotency = _records()
    updated = run.model_copy(
        update={
            "state": RunState.CLARIFYING,
            "revision": 2,
            "next_event_cursor": 3,
        }
    )
    next_event = event.model_copy(update={"cursor": 2, "event_type": "run.started"})

    await repository.commit(
        updated,
        next_event,
        expected_revision=1,
        idempotency=idempotency.model_copy(update={"key": "start-key"}),
    )

    run_put = table.client.transactions[0]["TransactItems"][0]["Put"]
    assert run_put["ConditionExpression"] == "revision = :revision AND owner_id = :owner"
    assert run_put["ExpressionAttributeValues"][":revision"] == {"N": "1"}
    assert run_put["ExpressionAttributeValues"][":owner"] == {"S": "user-1"}
