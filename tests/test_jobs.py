from datetime import UTC, datetime

import pytest

from deep_research.contracts.jobs import JobPhase, PhaseJob
from deep_research.jobs.memory import InMemoryJobDispatcher
from deep_research.jobs.sqs import SQSJobDispatcher


def _job() -> PhaseJob:
    return PhaseJob(
        phase=JobPhase.CLARIFY,
        run_id="a" * 32,
        tenant_id="tenant-1",
        owner_id="user-1",
        expected_revision=2,
        enqueued_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_memory_dispatcher_redelivers_unacknowledged_jobs() -> None:
    dispatcher = InMemoryJobDispatcher()
    await dispatcher.dispatch(_job())
    first = await dispatcher.receive(wait_seconds=1)
    assert first is not None and first.delivery_count == 1

    await dispatcher.redeliver_inflight()
    second = await dispatcher.receive(wait_seconds=1)

    assert second is not None
    assert second.job == first.job
    assert second.delivery_count == 2
    await dispatcher.acknowledge(second)


class FakeSQS:
    def __init__(self) -> None:
        self.sent = None
        self.deleted = None

    def send_message(self, **kwargs):
        self.sent = kwargs

    def receive_message(self, **kwargs):
        return {
            "Messages": [
                {
                    "Body": self.sent["MessageBody"],
                    "ReceiptHandle": "receipt-1",
                    "Attributes": {"ApproximateReceiveCount": "3"},
                }
            ]
        }

    def delete_message(self, **kwargs):
        self.deleted = kwargs


@pytest.mark.asyncio
async def test_sqs_adapter_round_trips_the_typed_protocol() -> None:
    client = FakeSQS()
    dispatcher = SQSJobDispatcher(client, "https://sqs.test/jobs")
    job = _job()

    await dispatcher.dispatch(job)
    delivery = await dispatcher.receive(wait_seconds=0)
    assert delivery is not None
    assert delivery.job == job
    assert delivery.delivery_count == 3

    await dispatcher.acknowledge(delivery)
    assert client.deleted["ReceiptHandle"] == "receipt-1"
