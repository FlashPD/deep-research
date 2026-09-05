import re
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from deep_research.contracts.research import WebSearchOperation, WebSearchResult

_EXACT_DATE_RANGE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*(?:/|\.\.|\bto\b)\s*(\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
_TAVILY_TIME_RANGES = {"day", "week", "month", "year", "d", "w", "m", "y"}


class _TavilyResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1, max_length=4_000)
    url: str = Field(pattern=r"^https?://", max_length=4_000)
    content: str | None = Field(default=None, max_length=10_000)


class _TavilyResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[_TavilyResult]


class TavilySearchAdapter:
    """Tavily discovery adapter. Returned snippets are never promoted directly to evidence."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.tavily.com",
        timeout_seconds: float = 20,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Tavily API key cannot be empty")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def search_web(self, operation: WebSearchOperation) -> list[WebSearchResult]:
        payload: dict[str, Any] = {
            "query": operation.query,
            "search_depth": "basic",
            "max_results": operation.max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "safe_search": True,
        }
        if operation.domains:
            payload["include_domains"] = operation.domains
        if operation.date_range:
            normalized = operation.date_range.casefold().strip()
            if normalized in _TAVILY_TIME_RANGES:
                payload["time_range"] = normalized
            elif match := _EXACT_DATE_RANGE.fullmatch(operation.date_range.strip()):
                payload["start_date"], payload["end_date"] = match.groups()

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            follow_redirects=False,
        )
        try:
            response = await client.post(
                "/search",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            parsed = _TavilyResponse.model_validate(response.json())
            return [
                WebSearchResult(
                    url=item.url,
                    title=item.title,
                    snippet=item.content,
                )
                for item in parsed.results[: operation.max_results]
            ]
        finally:
            if owns_client:
                await client.aclose()
