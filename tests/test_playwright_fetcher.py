"""Browser-free checks for the fetcher's request guard and timing defaults."""

import ipaddress

import pytest

from deep_research.tools.playwright_fetcher import (
    _DEFAULT_NAVIGATION_TIMEOUT_MS,
    _DEFAULT_SETTLE_TIMEOUT_MS,
    PublicUrlGuard,
    UnsafePublicUrlError,
    _should_block_resource,
)


class StubResolver:
    def __init__(self, address: str) -> None:
        self.calls: list[tuple[str, int]] = []
        self._address = ipaddress.ip_address(address)

    async def __call__(self, hostname: str, port: int):
        self.calls.append((hostname, port))
        return [self._address]


@pytest.mark.parametrize(
    ("resource_type", "blocked"),
    [("image", True), ("media", True), ("font", True), ("Image", True),
     ("document", False), ("script", False), ("xhr", False), ("stylesheet", False)],
)
def test_text_free_resources_are_blocked_before_dns(resource_type: str, blocked: bool) -> None:
    assert _should_block_resource(resource_type) is blocked


@pytest.mark.asyncio
async def test_guard_resolves_each_host_once_per_fetch() -> None:
    resolver = StubResolver("93.184.216.34")
    guard = PublicUrlGuard(resolver)

    await guard.ensure("https://example.com/review")
    await guard.ensure("https://example.com/static/app.js")
    await guard.ensure("https://EXAMPLE.com/api/data?x=1")
    await guard.ensure("https://cdn.example.com/bundle.js")

    assert resolver.calls == [("example.com", 443), ("cdn.example.com", 443)]
    assert guard.resolved_hosts == 2


@pytest.mark.asyncio
async def test_guard_caches_private_network_rejections() -> None:
    resolver = StubResolver("10.0.0.8")
    guard = PublicUrlGuard(resolver)

    with pytest.raises(UnsafePublicUrlError, match="non-public"):
        await guard.ensure("https://intranet.example.com/")
    with pytest.raises(UnsafePublicUrlError, match="non-public"):
        await guard.ensure("https://intranet.example.com/other")

    assert len(resolver.calls) == 1


@pytest.mark.asyncio
async def test_guard_rejects_bad_url_shapes_without_resolving() -> None:
    resolver = StubResolver("93.184.216.34")
    guard = PublicUrlGuard(resolver)

    for url in ("ftp://example.com/x", "https://user:pw@example.com/", "https://example.com:8443/",
                "http://localhost/", "https://box.local/"):
        with pytest.raises(UnsafePublicUrlError):
            await guard.ensure(url)

    assert resolver.calls == []


def test_navigation_plus_settle_fits_inside_the_research_tool_timeout() -> None:
    from deep_research.contracts.research import ResearchTask

    tool_timeout_ms = ResearchTask.model_fields["tool_timeout_seconds"].default * 1_000
    assert _DEFAULT_NAVIGATION_TIMEOUT_MS + _DEFAULT_SETTLE_TIMEOUT_MS < tool_timeout_ms
