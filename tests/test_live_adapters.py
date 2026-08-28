import json

import httpx
import pytest

from deep_research.contracts.research import WebSearchOperation
from deep_research.tools.playwright_fetcher import UnsafePublicUrlError, ensure_public_url
from deep_research.tools.tavily import TavilySearchAdapter


@pytest.mark.asyncio
async def test_tavily_adapter_sends_bounded_discovery_request() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Official statistics",
                        "url": "https://example.gov/statistics",
                        "content": "Discovery snippet only",
                        "score": 0.99,
                    }
                ]
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.tavily.com",
        transport=httpx.MockTransport(handler),
    )
    adapter = TavilySearchAdapter("secret", client=client)

    results = await adapter.search_web(
        WebSearchOperation(
            query="official market statistics",
            research_question_ids=["market_size"],
            date_range="2025-01-01 to 2025-12-31",
            domains=["example.gov"],
            max_results=3,
        )
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert captured["authorization"] == "Bearer secret"
    assert payload["start_date"] == "2025-01-01"
    assert payload["end_date"] == "2025-12-31"
    assert payload["include_raw_content"] is False
    assert payload["safe_search"] is True
    assert results[0].snippet == "Discovery snippet only"
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1/private",
        "http://user:password@8.8.8.8/",
        "http://8.8.8.8:8080/",
    ],
)
async def test_playwright_url_guard_rejects_private_or_unsafe_targets(url: str) -> None:
    with pytest.raises(UnsafePublicUrlError):
        await ensure_public_url(url)


@pytest.mark.asyncio
async def test_playwright_url_guard_accepts_public_https_ip() -> None:
    await ensure_public_url("https://8.8.8.8/")
