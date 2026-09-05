import asyncio

from deep_research.contracts.research import FetchedPage, FetchPageRequest
from deep_research.tools.research import PageFetchAdapter
from deep_research.tools.urls import canonicalize_url


class CachingPageFetcher:
    """Single-flight page cache so parallel workstreams share one capture per URL.

    Two fetches of a dynamic page can return different text, which would give the same
    source ID two different content hashes. Sharing one capture per canonical URL within a
    research phase keeps source records identical across workstreams and avoids paying for
    the same page twice. Failures are cached as well so a timed-out URL is not retried by
    every workstream that discovered it.
    """

    def __init__(self, inner: PageFetchAdapter) -> None:
        self._inner = inner
        self._inflight: dict[str, asyncio.Future[FetchedPage]] = {}

    async def fetch_page(self, request: FetchPageRequest) -> FetchedPage:
        key = canonicalize_url(request.url)
        future = self._inflight.get(key)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._inflight[key] = future
            try:
                page = await self._inner.fetch_page(request)
            except Exception as exc:
                _fail_future(future, exc)
                raise
            except BaseException:
                # Cancellation (for example the research tool timeout) must not propagate
                # as a cancellation into other workstreams awaiting this capture. They
                # receive an ordinary error and the entry is released for a later retry.
                _fail_future(
                    future,
                    RuntimeError("the shared fetch of this page was cancelled before completing"),
                )
                self._inflight.pop(key, None)
                raise
            future.set_result(page)
            return page
        return await asyncio.shield(future)

    @property
    def cached_urls(self) -> int:
        return len(self._inflight)


def _fail_future(future: asyncio.Future[FetchedPage], exc: BaseException) -> None:
    future.set_exception(exc)
    # The owner re-raises to its own caller, so the shared future may never be awaited by a
    # waiter. Observe the exception here so asyncio does not log "Future exception was never
    # retrieved" when the cache is garbage-collected.
    future.exception()
