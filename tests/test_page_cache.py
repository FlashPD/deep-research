import asyncio

import pytest

from deep_research.contracts.research import FetchedPage, FetchPageRequest
from deep_research.tools.page_cache import CachingPageFetcher


class CountingFetcher:
    """Returns different content on every call, like a dynamic page would."""

    def __init__(self, gate: asyncio.Event | None = None, error: Exception | None = None):
        self.calls = 0
        self._gate = gate
        self._error = error

    async def fetch_page(self, request: FetchPageRequest) -> FetchedPage:
        self.calls += 1
        if self._gate is not None:
            await self._gate.wait()
        if self._error is not None:
            raise self._error
        return FetchedPage(
            final_url=request.url,
            title="Dynamic page",
            content=f"capture number {self.calls} " + "x" * 300,
            access_date="2026-09-05",
        )


@pytest.mark.asyncio
async def test_concurrent_fetches_of_one_canonical_url_share_a_single_capture() -> None:
    inner = CountingFetcher(gate=(gate := asyncio.Event()))
    cache = CachingPageFetcher(inner)

    first = asyncio.create_task(
        cache.fetch_page(FetchPageRequest(url="https://example.com/review?utm_source=news"))
    )
    second = asyncio.create_task(
        cache.fetch_page(FetchPageRequest(url="https://example.com/review"))
    )
    await asyncio.sleep(0)
    gate.set()
    pages = await asyncio.gather(first, second)

    assert inner.calls == 1
    assert pages[0].content == pages[1].content
    assert cache.cached_urls == 1


@pytest.mark.asyncio
async def test_failed_capture_is_shared_and_not_refetched_within_the_phase() -> None:
    inner = CountingFetcher(error=RuntimeError("page returned HTTP 403"))
    cache = CachingPageFetcher(inner)
    request = FetchPageRequest(url="https://example.com/blocked")

    with pytest.raises(RuntimeError, match="403"):
        await cache.fetch_page(request)
    with pytest.raises(RuntimeError, match="403"):
        await cache.fetch_page(request)

    assert inner.calls == 1


@pytest.mark.asyncio
async def test_unawaited_failed_capture_does_not_log_an_unretrieved_exception() -> None:
    import gc

    loop = asyncio.get_running_loop()
    reports: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reports.append(context))
    try:
        cache = CachingPageFetcher(CountingFetcher(error=RuntimeError("HTTP 403")))
        with pytest.raises(RuntimeError):
            await cache.fetch_page(FetchPageRequest(url="https://example.com/only-once"))
        # Nobody else waits on the shared future; dropping the cache finalizes it.
        del cache
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert not [item for item in reports if "never retrieved" in item.get("message", "")]


@pytest.mark.asyncio
async def test_cancelled_owner_does_not_cancel_waiters_and_releases_the_entry() -> None:
    gate = asyncio.Event()
    inner = CountingFetcher(gate=gate)
    cache = CachingPageFetcher(inner)
    request = FetchPageRequest(url="https://example.com/slow")

    owner = asyncio.create_task(asyncio.wait_for(cache.fetch_page(request), timeout=0.05))
    await asyncio.sleep(0)
    waiter = asyncio.create_task(cache.fetch_page(request))

    with pytest.raises(TimeoutError):
        await owner
    with pytest.raises(RuntimeError, match="cancelled"):
        await waiter

    gate.set()
    page = await cache.fetch_page(request)

    assert page.content.startswith("capture number 2")
    assert inner.calls == 2
