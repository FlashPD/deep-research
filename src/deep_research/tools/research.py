from typing import Protocol

from deep_research.contracts.research import (
    FetchedPage,
    FetchPageRequest,
    UploadChunk,
    UploadSearchOperation,
    WebSearchOperation,
    WebSearchResult,
)


class WebSearchAdapter(Protocol):
    async def search_web(self, operation: WebSearchOperation) -> list[WebSearchResult]: ...


class PageFetchAdapter(Protocol):
    async def fetch_page(self, request: FetchPageRequest) -> FetchedPage: ...


class UploadSearchAdapter(Protocol):
    async def search_uploads(self, operation: UploadSearchOperation) -> list[UploadChunk]: ...


class ResearchAdapter(WebSearchAdapter, PageFetchAdapter, UploadSearchAdapter, Protocol):
    """Run-bound research capabilities.

    Implementations obtain tenant and run identity when they are constructed. Agent-authored
    operations therefore cannot select a tenant, run, index, browser session, or credential.
    """

    async def search_web(self, operation: WebSearchOperation) -> list[WebSearchResult]: ...

    async def fetch_page(self, request: FetchPageRequest) -> FetchedPage: ...

    async def search_uploads(self, operation: UploadSearchOperation) -> list[UploadChunk]: ...


class ComposedResearchAdapter:
    """Combines independently deployable live adapters behind the agent-facing protocol."""

    def __init__(
        self,
        web_search: WebSearchAdapter,
        page_fetcher: PageFetchAdapter,
        upload_search: UploadSearchAdapter,
    ) -> None:
        self._web_search = web_search
        self._page_fetcher = page_fetcher
        self._upload_search = upload_search

    async def search_web(self, operation: WebSearchOperation) -> list[WebSearchResult]:
        return await self._web_search.search_web(operation)

    async def fetch_page(self, request: FetchPageRequest) -> FetchedPage:
        return await self._page_fetcher.fetch_page(request)

    async def search_uploads(self, operation: UploadSearchOperation) -> list[UploadChunk]:
        return await self._upload_search.search_uploads(operation)


class UnconfiguredResearchAdapter:
    """Safe default used until a run-scoped live or local adapter is injected."""

    @staticmethod
    def _unavailable() -> RuntimeError:
        return RuntimeError("no research adapter is configured for this application")

    async def search_web(self, operation: WebSearchOperation) -> list[WebSearchResult]:
        raise self._unavailable()

    async def fetch_page(self, request: FetchPageRequest) -> FetchedPage:
        raise self._unavailable()

    async def search_uploads(self, operation: UploadSearchOperation) -> list[UploadChunk]:
        raise self._unavailable()
