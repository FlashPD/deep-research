import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from io import BytesIO
from typing import Any
from urllib.parse import urlsplit

from deep_research.contracts.evidence import SourceType
from deep_research.contracts.research import FetchedPage, FetchPageRequest

_ALLOWED_PORTS = {None, 80, 443}
_ALLOWED_SCHEMES = {"http", "https"}
_PDF_MEDIA_TYPE = "application/pdf"
# Sub-resources that never contribute extractable text. Aborting them before any DNS work
# makes page loads faster and lets `networkidle` settle sooner.
_BLOCKED_RESOURCE_TYPES = frozenset({"image", "media", "font"})
# Default navigation + settle budget stays below the 30-second research tool timeout so the
# fetcher's own error (not a bare cancellation) is what the research limitation records.
_DEFAULT_NAVIGATION_TIMEOUT_MS = 20_000
_DEFAULT_SETTLE_TIMEOUT_MS = 5_000

Resolver = Callable[[str, int], Awaitable[list[ipaddress.IPv4Address | ipaddress.IPv6Address]]]


class UnsafePublicUrlError(ValueError):
    """Raised when browser navigation could reach a non-public network target."""


def _should_block_resource(resource_type: str) -> bool:
    return resource_type.casefold() in _BLOCKED_RESOURCE_TYPES


def _validate_url_shape(url: str) -> tuple[str, int]:
    """Cheap syntactic SSRF checks; returns (hostname, port) for the network check."""
    parsed = urlsplit(url)
    if parsed.scheme.casefold() not in _ALLOWED_SCHEMES or not parsed.hostname:
        raise UnsafePublicUrlError("only absolute HTTP(S) URLs are allowed")
    if parsed.username or parsed.password:
        raise UnsafePublicUrlError("URLs containing credentials are prohibited")
    if parsed.port not in _ALLOWED_PORTS:
        raise UnsafePublicUrlError("non-standard URL ports are prohibited")
    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise UnsafePublicUrlError("local hostnames are prohibited")
    port = parsed.port or (443 if parsed.scheme.casefold() == "https" else 80)
    return hostname, port


async def _resolve_addresses(
    hostname: str, port: int
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        return [ipaddress.ip_address(hostname)]
    except ValueError:
        records = await asyncio.to_thread(
            socket.getaddrinfo, hostname, port, type=socket.SOCK_STREAM
        )
        return list({ipaddress.ip_address(record[4][0]) for record in records})


async def _ensure_public_addresses(
    hostname: str, port: int, resolver: Resolver
) -> None:
    addresses = await resolver(hostname, port)
    if not addresses or any(not address.is_global for address in addresses):
        raise UnsafePublicUrlError("URL resolves to a non-public network address")


async def ensure_public_url(url: str) -> None:
    hostname, port = _validate_url_shape(url)
    await _ensure_public_addresses(hostname, port, _resolve_addresses)


class PublicUrlGuard:
    """SSRF guard that resolves each host once per fetch.

    A rendered page issues dozens of sub-resource requests to a handful of hosts. Without a
    cache every request paid for a threaded DNS lookup, which was enough to push slow pages
    past the navigation timeout.
    """

    def __init__(self, resolver: Resolver | None = None) -> None:
        self._resolver = resolver or _resolve_addresses
        self._verdicts: dict[tuple[str, int], UnsafePublicUrlError | None] = {}
        self._pending: dict[tuple[str, int], asyncio.Future[None]] = {}

    async def ensure(self, url: str) -> None:
        hostname, port = _validate_url_shape(url)
        key = (hostname, port)
        if key in self._verdicts:
            verdict = self._verdicts[key]
            if verdict is not None:
                raise verdict
            return
        pending = self._pending.get(key)
        if pending is not None:
            await asyncio.shield(pending)
            return await self.ensure(url)
        future = asyncio.get_running_loop().create_future()
        self._pending[key] = future
        try:
            await _ensure_public_addresses(hostname, port, self._resolver)
        except UnsafePublicUrlError as exc:
            self._verdicts[key] = exc
            raise
        else:
            self._verdicts[key] = None
        finally:
            self._pending.pop(key, None)
            if not future.done():
                future.set_result(None)

    @property
    def resolved_hosts(self) -> int:
        return len(self._verdicts)


class PlaywrightPageFetcher:
    """Isolated browser fetcher with SSRF checks and bounded HTML/PDF extraction."""

    def __init__(
        self,
        *,
        headless: bool = True,
        navigation_timeout_ms: int = _DEFAULT_NAVIGATION_TIMEOUT_MS,
        settle_timeout_ms: int = _DEFAULT_SETTLE_TIMEOUT_MS,
        max_redirects: int = 5,
        max_response_bytes: int = 10 * 1024 * 1024,
        max_content_chars: int = 100_000,
        max_pdf_pages: int = 500,
        browser_launch_options: dict[str, Any] | None = None,
    ) -> None:
        self._headless = headless
        self._navigation_timeout_ms = navigation_timeout_ms
        self._settle_timeout_ms = settle_timeout_ms
        self._max_redirects = max_redirects
        self._max_response_bytes = max_response_bytes
        self._max_content_chars = max_content_chars
        self._max_pdf_pages = max_pdf_pages
        self._browser_launch_options = browser_launch_options or {}

    async def fetch_page(self, request: FetchPageRequest) -> FetchedPage:
        guard = PublicUrlGuard()
        await guard.ensure(request.url)
        try:
            from playwright.async_api import TimeoutError as PlaywrightTimeoutError
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - depends on optional browser installation
            raise RuntimeError(
                "Playwright is not installed; install project dependencies and run "
                "`playwright install chromium`"
            ) from exc

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=self._headless,
                **self._browser_launch_options,
            )
            context = await browser.new_context(
                java_script_enabled=True,
                service_workers="block",
                accept_downloads=False,
            )

            async def guard_request(route: Any) -> None:
                if _should_block_resource(route.request.resource_type):
                    await route.abort("blockedbyclient")
                    return
                target = route.request.url
                if urlsplit(target).scheme.casefold() not in _ALLOWED_SCHEMES:
                    await route.abort("blockedbyclient")
                    return
                try:
                    await guard.ensure(target)
                except (UnsafePublicUrlError, OSError, ValueError):
                    await route.abort("blockedbyclient")
                    return
                await route.continue_()

            await context.route("**/*", guard_request)
            page = await context.new_page()
            page.set_default_timeout(self._navigation_timeout_ms)
            try:
                response = await page.goto(
                    request.url,
                    wait_until="domcontentloaded",
                    timeout=self._navigation_timeout_ms,
                )
                if response is None:
                    raise RuntimeError("browser navigation returned no response")
                if response.status >= 400:
                    raise RuntimeError(f"page returned HTTP {response.status}")
                await guard.ensure(response.url)
                if _redirect_count(response.request) > self._max_redirects:
                    raise RuntimeError("page exceeded the redirect ceiling")
                # Let client-side rendering settle so hydrated text is captured consistently.
                # A busy page (analytics beacons, long polling) simply proceeds after the cap.
                try:
                    await page.wait_for_load_state(
                        "networkidle", timeout=self._settle_timeout_ms
                    )
                except PlaywrightTimeoutError:
                    pass

                content_length = await response.header_value("content-length")
                if content_length and int(content_length) > self._max_response_bytes:
                    raise RuntimeError("page exceeded the response-size ceiling")
                content_type = (
                    (await response.header_value("content-type")) or ""
                ).casefold()
                access_date = datetime.now(UTC).date().isoformat()
                is_pdf_url = urlsplit(response.url).path.casefold().endswith(".pdf")
                if _PDF_MEDIA_TYPE in content_type or is_pdf_url:
                    body = await response.body()
                    if len(body) > self._max_response_bytes:
                        raise RuntimeError("PDF exceeded the response-size ceiling")
                    text = await asyncio.to_thread(
                        _extract_pdf_text, body, self._max_pdf_pages
                    )
                    title = _filename_from_url(response.url) or "Public PDF"
                    source_type = SourceType.PUBLIC_PDF
                    location = "Extracted PDF text"
                else:
                    text = await page.locator("body").inner_text()
                    text = _normalize_page_text(text)
                    title = (await page.title()).strip() or response.url
                    source_type = SourceType.WEB_PAGE
                    location = "Rendered page text"
                if len(text) > self._max_content_chars:
                    raise RuntimeError("extracted page text exceeded the content ceiling")
                if not text.strip():
                    raise RuntimeError("page contained no extractable text")

                return FetchedPage(
                    final_url=response.url,
                    title=title,
                    content=text,
                    source_type=source_type,
                    publisher=await _meta_content(page, "meta[property='og:site_name']"),
                    author=await _meta_content(page, "meta[name='author']"),
                    publication_date=await _first_meta_content(
                        page,
                        [
                            "meta[property='article:published_time']",
                            "meta[name='date']",
                            "meta[name='publication_date']",
                        ],
                    ),
                    access_date=access_date,
                    location=location,
                )
            finally:
                # Route handlers run in background tasks. Detach them before closing the
                # context so an in-flight continue_/abort does not race with shutdown and
                # emit an unhandled TargetClosedError (especially when the caller cancels
                # this fetch at its timeout).
                await context.unroute_all(behavior="ignoreErrors")
                await context.close()
                await browser.close()


def _redirect_count(request: Any) -> int:
    count = 0
    current = request.redirected_from
    while current is not None:
        count += 1
        current = current.redirected_from
    return count


async def _meta_content(page: Any, selector: str) -> str | None:
    locator = page.locator(selector).first
    if await locator.count() == 0:
        return None
    value = await locator.get_attribute("content")
    return value.strip()[:500] if value and value.strip() else None


async def _first_meta_content(page: Any, selectors: list[str]) -> str | None:
    for selector in selectors:
        if value := await _meta_content(page, selector):
            return value[:100]
    return None


def _extract_pdf_text(body: bytes, max_pages: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(body))
    if reader.is_encrypted:
        raise RuntimeError("encrypted public PDFs are not supported")
    if len(reader.pages) > max_pages:
        raise RuntimeError("PDF exceeded the page-count ceiling")
    return "\n\n".join(
        f"Page {number}\n{page.extract_text() or ''}"
        for number, page in enumerate(reader.pages, start=1)
    ).strip()


def _filename_from_url(url: str) -> str | None:
    filename = urlsplit(url).path.rsplit("/", 1)[-1]
    return filename or None


def _normalize_page_text(value: str) -> str:
    value = value.replace("\x00", "")
    return re.sub(r"\n{3,}", "\n\n", value).strip()
