from dataclasses import dataclass

from deep_research.settings import AppSettings
from deep_research.tools.playwright_fetcher import PlaywrightPageFetcher
from deep_research.tools.research import ComposedResearchAdapter
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
    upload_ingestion: UploadIngestionService


def build_live_research_services(
    settings: AppSettings,
    *,
    tenant_id: str,
    run_id: str,
) -> LiveResearchServices:
    """Create live services bound to one authorized tenant/run invocation."""

    if settings.tavily_api_key is None:
        raise ValueError("TAVILY_API_KEY is required for live research")
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
    research_adapter = ComposedResearchAdapter(
        web_search=TavilySearchAdapter(
            settings.tavily_api_key.get_secret_value(),
            base_url=settings.tavily_base_url,
        ),
        page_fetcher=PlaywrightPageFetcher(headless=settings.playwright_headless),
        upload_search=UploadIndexSearchAdapter(upload_index),
    )
    upload_ingestion = UploadIngestionService(
        index=upload_index,
        artifact_store=LocalUploadArtifactStore(
            settings.upload_artifact_root,
            tenant_id=tenant_id,
            run_id=run_id,
        ),
        malware_scanner=ClamAVScanner(settings.clamav_host, settings.clamav_port),
    )
    return LiveResearchServices(
        research_adapter=research_adapter,
        upload_ingestion=upload_ingestion,
    )
