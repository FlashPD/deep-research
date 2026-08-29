import asyncio
from typing import Any

from deep_research.contracts.jobs import JobDelivery, PhaseJob


class SQSJobDispatcher:
    """Thin async adapter over an SQS client; Standard queues are intentionally supported."""

    def __init__(self, client: Any, queue_url: str) -> None:
        if not queue_url:
            raise ValueError("queue_url is required")
        self._client = client
        self._queue_url = queue_url

    async def dispatch(self, job: PhaseJob) -> None:
        kwargs: dict[str, Any] = {
            "QueueUrl": self._queue_url,
            "MessageBody": job.model_dump_json(),
        }
        if self._queue_url.endswith(".fifo"):
            kwargs.update(
                MessageGroupId=job.run_id,
                MessageDeduplicationId=str(job.job_id),
            )
        await asyncio.to_thread(self._client.send_message, **kwargs)

    async def receive(self, *, wait_seconds: int = 20) -> JobDelivery | None:
        response = await asyncio.to_thread(
            self._client.receive_message,
            QueueUrl=self._queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=min(max(wait_seconds, 0), 20),
            AttributeNames=["ApproximateReceiveCount"],
        )
        messages = response.get("Messages", [])
        if not messages:
            return None
        message = messages[0]
        return JobDelivery(
            job=PhaseJob.model_validate_json(message["Body"]),
            receipt_handle=message["ReceiptHandle"],
            delivery_count=int(message.get("Attributes", {}).get("ApproximateReceiveCount", "1")),
        )

    async def acknowledge(self, delivery: JobDelivery) -> None:
        await asyncio.to_thread(
            self._client.delete_message,
            QueueUrl=self._queue_url,
            ReceiptHandle=delivery.receipt_handle,
        )

    async def retry(self, delivery: JobDelivery, *, delay_seconds: int = 0) -> None:
        await asyncio.to_thread(
            self._client.change_message_visibility,
            QueueUrl=self._queue_url,
            ReceiptHandle=delivery.receipt_handle,
            VisibilityTimeout=min(max(delay_seconds, 0), 43_200),
        )
