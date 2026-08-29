from collections.abc import Callable
from dataclasses import dataclass

from deep_research.contracts.research import UploadChunk, UploadSearchOperation
from deep_research.settings import AppSettings
from deep_research.tools.playwright_fetcher import PlaywrightPageFetcher
from deep_research.tools.research import (
    ComposedResearchAdapter,
    UnconfiguredUploadSearchAdapter,
    UploadSearchAdapter,
)
from deep_research.tools.tavily import TavilySearchAdapter
from deep_research.uploads.index import OpenSearchUploadIndex, UploadIndexSearchAdapter
from deep_research.uploads.ingestion import (
    ClamAVScanner,
    LocalUploadArtifactStore,
    UploadIngestionService,
)


@dataclass(frozen=True)
class LiveResearchServices:
    research_adapter: ComposedResearchAdapter
    upload_ingestion: UploadIngestionService | None = None


class LazyUploadSearchAdapter:
    """Construct upload infrastructure only if an approved operation actually uses it."""

    def __init__(self, factory: Callable[[], UploadSearchAdapter]) -> None:
        self._factory = factory
        self._adapter: UploadSearchAdapter | None = None

    async def search_uploads(self, operation: UploadSearchOperation) -> list[UploadChunk]:
        if self._adapter is None:
            self._adapter = self._factory()
        return await self._adapter.search_uploads(operation)


def build_live_research_services(
    settings: AppSettings,
    *,
    tenant_id: str,
    run_id: str,
) -> LiveResearchServices:
    """Create live services bound to one authorized tenant/run invocation."""

    if (
        settings.tavily_api_key is None
        or not settings.tavily_api_key.get_secret_value().strip()
    ):
        raise ValueError("TAVILY_API_KEY is required for live research")
    upload_search: UploadSearchAdapter = UnconfiguredUploadSearchAdapter()
    if settings.uploads_enabled:
        upload_search = LazyUploadSearchAdapter(
            lambda: _build_upload_index(settings, tenant_id=tenant_id, run_id=run_id)
        )
    research_adapter = ComposedResearchAdapter(
        web_search=TavilySearchAdapter(
            settings.tavily_api_key.get_secret_value(),
            base_url=settings.tavily_base_url,
        ),
        page_fetcher=PlaywrightPageFetcher(headless=settings.playwright_headless),
        upload_search=upload_search,
    )
    return LiveResearchServices(research_adapter=research_adapter)


def _build_upload_index(
    settings: AppSettings, *, tenant_id: str, run_id: str
) -> UploadIndexSearchAdapter:
    upload_index = OpenSearchUploadIndex.from_url(
        url=settings.opensearch_url,
        tenant_id=tenant_id,
        run_id=run_id,
        index_name=settings.opensearch_index,
        username=settings.opensearch_username,
        password=(
            settings.opensearch_password.get_secret_value()
            if settings.opensearch_password
            else None
        ),
        aws_region=settings.opensearch_aws_region,
        aws_service=settings.opensearch_aws_service,
    )
    return UploadIndexSearchAdapter(upload_index)


def build_upload_ingestion_service(
    settings: AppSettings, *, tenant_id: str, run_id: str
) -> UploadIngestionService:
    """Explicit upload-only composition; never called by a public-web run."""
    if not settings.uploads_enabled:
        raise ValueError("UPLOADS_ENABLED=true is required for upload ingestion")
    upload_index = OpenSearchUploadIndex.from_url(
        url=settings.opensearch_url,
        tenant_id=tenant_id,
        run_id=run_id,
        index_name=settings.opensearch_index,
        username=settings.opensearch_username,
        password=(
            settings.opensearch_password.get_secret_value()
            if settings.opensearch_password
            else None
        ),
        aws_region=settings.opensearch_aws_region,
        aws_service=settings.opensearch_aws_service,
    )
    return UploadIngestionService(
        index=upload_index,
        artifact_store=LocalUploadArtifactStore(
            settings.upload_artifact_root,
            tenant_id=tenant_id,
            run_id=run_id,
        ),
        malware_scanner=ClamAVScanner(settings.clamav_host, settings.clamav_port),
    )
