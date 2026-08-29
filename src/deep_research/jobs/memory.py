import asyncio
from collections import defaultdict
from uuid import uuid4

from deep_research.contracts.jobs import JobDelivery, PhaseJob


class InMemoryJobDispatcher:
    """Local adapter with explicit ack/retry semantics matching the worker contract."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[PhaseJob, int]] = asyncio.Queue()
        self._inflight: dict[str, tuple[PhaseJob, int]] = {}
        self._deliveries: defaultdict[str, int] = defaultdict(int)

    async def dispatch(self, job: PhaseJob) -> None:
        await self._queue.put((job.model_copy(deep=True), 0))

    async def receive(self, *, wait_seconds: int = 20) -> JobDelivery | None:
        try:
            job, prior_count = await asyncio.wait_for(
                self._queue.get(), timeout=max(0.001, wait_seconds)
            )
        except TimeoutError:
            return None
        receipt = uuid4().hex
        count = max(prior_count + 1, self._deliveries[str(job.job_id)] + 1)
        self._deliveries[str(job.job_id)] = count
        self._inflight[receipt] = (job, count)
        return JobDelivery(job=job, receipt_handle=receipt, delivery_count=count)

    async def acknowledge(self, delivery: JobDelivery) -> None:
        self._inflight.pop(delivery.receipt_handle, None)

    async def retry(self, delivery: JobDelivery, *, delay_seconds: int = 0) -> None:
        item = self._inflight.pop(delivery.receipt_handle, None)
        if item is None:
            return
        job, count = item
        if delay_seconds:
            await asyncio.sleep(delay_seconds)
        await self._queue.put((job, count))

    async def redeliver_inflight(self) -> None:
        """Test/local crash simulation: return all unacked messages to the queue."""
        items = list(self._inflight.values())
        self._inflight.clear()
        for item in items:
            await self._queue.put(item)

    @property
    def pending(self) -> int:
        return self._queue.qsize()
