
import pytest
from pydantic import ValidationError

from deep_research.agents.researcher import (
    ResearchAgent,
    _segment_materials,
    canonicalize_url,
    merge_research_results,
)
from deep_research.contracts.evidence import BudgetUsage, EvidenceRepairTask, SupportStrength
from deep_research.contracts.planning import DepthPreset
from deep_research.contracts.research import (
    DraftEvidenceSelection,
    DraftResearchClaim,
    FetchedPage,
    FetchPageRequest,
    ResearchExecutionPlan,
    ResearchRequest,
    ResearchSynthesisDraft,
    ResearchTask,
    ResearchTaskStatus,
    UploadChunk,
    UploadSearchOperation,
    WebSearchOperation,
    WebSearchResult,
)
from tests.conftest import FakeGateway
from tests.factories import make_plan


class FakeResearchAdapter:
    def __init__(
        self,
        *,
        search_results: list[WebSearchResult] | None = None,
        page: FetchedPage | BaseException | None = None,
        upload_chunks: list[UploadChunk] | None = None,
    ) -> None:
        self.search_results = search_results or []
        self.page = page
        self.upload_chunks = upload_chunks or []
        self.web_calls: list[WebSearchOperation] = []
        self.fetch_calls: list[FetchPageRequest] = []
        self.upload_calls: list[UploadSearchOperation] = []

    async def search_web(self, operation: WebSearchOperation) -> list[WebSearchResult]:
        self.web_calls.append(operation)
        return self.search_results

    async def fetch_page(self, request: FetchPageRequest) -> FetchedPage:
        self.fetch_calls.append(request)
        if isinstance(self.page, BaseException):
            raise self.page
        if self.page is None:
            raise RuntimeError("no page configured")
        return self.page

    async def search_uploads(self, operation: UploadSearchOperation) -> list[UploadChunk]:
        self.upload_calls.append(operation)
        return self.upload_chunks


def make_execution_plan() -> ResearchExecutionPlan:
    return ResearchExecutionPlan(
        web_searches=[
            WebSearchOperation(
                query="official EV market statistics",
                research_question_ids=["market_size"],
                max_results=5,
            )
        ],
        rationale="Use primary statistics to answer the approved market-size question.",
    )


@pytest.mark.asyncio
async def test_public_researcher_executes_approved_queries_without_model_planning() -> None:
    page = make_page()
    gateway = FakeGateway(make_synthesis(page))
    adapter = FakeResearchAdapter(
        search_results=[WebSearchResult(url="https://example.gov/ev", title="EV")],
        page=page,
    )

    result = await ResearchAgent(gateway, adapter).research(make_request())

    assert len(gateway.calls) == 1
    assert "research-synthesis" in gateway.calls[0]["prompt"]
    assert len(adapter.web_calls) == 1
    assert adapter.web_calls[0].query == "market query 0"
    assert result.evidence.sources


def make_page() -> FetchedPage:
    return FetchedPage(
        final_url="https://example.gov/ev?utm_source=newsletter",
        title="Official EV statistics",
        content=(
            "Official records show that EV sales increased by 20 percent in 2025. "
            "The published table covers registrations in every reporting region and explains "
            "the collection methodology, revisions, exclusions, and comparison with 2024. "
            "The agency reports the result as final rather than a preliminary estimate."
        ),
        publisher="National Statistics Office",
        publication_date="2026-01-15",
        access_date="2026-08-27",
        location="Results, paragraph 2",
    )


def make_synthesis(
    page: FetchedPage,
    *,
    segment_id: str | None = None,
) -> ResearchSynthesisDraft:
    material = ResearchAgent._material_from_page(page)
    segment = _segment_materials([material])[0]
    return ResearchSynthesisDraft(
        claims=[
            DraftResearchClaim(
                research_question_id="market_size",
                section_ids=["market_findings"],
                normalized_claim="EV sales increased by 20 percent in 2025.",
                evidence=[
                    DraftEvidenceSelection(
                        material_id=material.material_id,
                        segment_id=segment_id or segment.segment_id,
                    )
                ],
                support_strength=SupportStrength.STRONG,
            )
        ]
    )


def make_request(*, uses_uploads: bool = False) -> ResearchRequest:
    plan = make_plan(uses_uploads=uses_uploads, query_count=1)
    return ResearchRequest(
        plan=plan,
        task=ResearchTask.for_workstream(plan, "market_analysis"),
    )


@pytest.mark.asyncio
async def test_researcher_fetches_discovery_results_and_records_typed_evidence() -> None:
    page = make_page()
    gateway = FakeGateway(make_synthesis(page))
    adapter = FakeResearchAdapter(
        search_results=[
            WebSearchResult(
                url="https://example.gov/ev?utm_campaign=test",
                title="Search result",
                snippet="This discovery snippet is not evidence.",
            )
        ],
        page=page,
    )

    result = await ResearchAgent(gateway, adapter).research(make_request())

    assert result.status is ResearchTaskStatus.COMPLETED
    assert len(adapter.web_calls) == 1
    assert len(adapter.fetch_calls) == 1
    assert len(result.evidence.sources) == 1
    assert result.evidence.sources[0].canonical_url == "https://example.gov/ev"
    assert "EV sales increased" in result.evidence.excerpts[0].excerpt
    assert "discovery snippet" not in result.model_dump_json()
    assert result.budget_usage.searches == 1
    assert result.budget_usage.model_calls == 1


@pytest.mark.asyncio
async def test_researcher_deduplicates_canonical_urls_before_fetching() -> None:
    page = make_page()
    adapter = FakeResearchAdapter(
        search_results=[
            WebSearchResult(url="https://example.gov/ev?utm_source=a", title="First"),
            WebSearchResult(url="https://EXAMPLE.gov/ev?utm_source=b", title="Duplicate"),
        ],
        page=page,
    )

    await ResearchAgent(
        FakeGateway(make_synthesis(page)), adapter
    ).research(make_request())

    assert len(adapter.fetch_calls) == 1


@pytest.mark.asyncio
async def test_researcher_tries_later_candidates_when_initial_fetch_fails() -> None:
    page = make_page()
    plan = make_plan(query_count=1)
    request = ResearchRequest(
        plan=plan,
        task=ResearchTask.for_workstream(plan, "market_analysis", max_sources=1),
    )

    class FallbackResearchAdapter(FakeResearchAdapter):
        async def fetch_page(self, request: FetchPageRequest) -> FetchedPage:
            self.fetch_calls.append(request)
            if len(self.fetch_calls) == 1:
                raise TimeoutError("first candidate timed out")
            return page

    adapter = FallbackResearchAdapter(
        search_results=[
            WebSearchResult(url="https://unavailable.example/ev", title="Unavailable"),
            WebSearchResult(url="https://example.gov/ev", title="Official EV statistics"),
        ],
    )

    result = await ResearchAgent(
        FakeGateway(make_synthesis(page)), adapter
    ).research(request)

    assert len(adapter.fetch_calls) == 2
    assert result.evidence.sources
    assert any(
        "Page fetch failed for unavailable.example: TimeoutError: first candidate timed out"
        in item
        for item in result.limitations
    )


def test_researcher_rejects_unpermitted_upload_operation() -> None:
    invalid = ResearchExecutionPlan(
        upload_searches=[
            UploadSearchOperation(
                query="private EV estimate",
                research_question_ids=["market_size"],
            )
        ],
        rationale="Search uploads.",
    )
    with pytest.raises(ValueError, match="unpermitted upload"):
        ResearchAgent._validate_execution_plan(invalid, make_request())


@pytest.mark.asyncio
async def test_researcher_repairs_excerpt_not_found_in_captured_material() -> None:
    page = make_page()
    invalid = make_synthesis(page, segment_id="G999")
    gateway = FakeGateway(invalid, make_synthesis(page))
    adapter = FakeResearchAdapter(
        search_results=[WebSearchResult(url="https://example.gov/ev", title="EV")],
        page=page,
    )

    result = await ResearchAgent(gateway, adapter).research(make_request())

    assert len(gateway.calls) == 2
    assert "unknown evidence segment" in gateway.calls[1]["prompt"]
    assert result.evidence.claims[0].evidence_ids


@pytest.mark.asyncio
async def test_search_result_is_not_evidence_when_fetch_fails() -> None:
    adapter = FakeResearchAdapter(
        search_results=[
            WebSearchResult(
                url="https://example.gov/ev",
                title="Discovery only",
                snippet="A claim that must not be recorded.",
            )
        ],
        page=TimeoutError("fetch failed"),
    )
    gateway = FakeGateway()

    result = await ResearchAgent(gateway, adapter).research(make_request())

    assert result.evidence.sources == []
    assert result.evidence.claims == []
    assert len(gateway.calls) == 0
    assert result.budget_usage.model_calls == 0
    assert any(
        "Page fetch failed for example.gov: TimeoutError: fetch failed" in item
        for item in result.limitations
    )


@pytest.mark.asyncio
async def test_shell_page_is_rejected_and_does_not_consume_source_budget() -> None:
    page = FetchedPage(
        final_url="https://www.youtube.com/watch?v=empty",
        title="TV comparison video",
        content="0:00 / 8:47",
        publisher="YouTube",
        access_date="2026-08-29",
    )
    adapter = FakeResearchAdapter(
        search_results=[WebSearchResult(url=page.final_url, title=page.title)],
        page=page,
    )

    result = await ResearchAgent(FakeGateway(), adapter).research(make_request())

    assert result.evidence.sources == []
    assert result.budget_usage.fetched_sources == 0
    assert any("Rejected unusable page" in item for item in result.limitations)


@pytest.mark.asyncio
async def test_repair_can_use_adaptive_sixth_quick_search() -> None:
    plan = make_plan(preset=DepthPreset.QUICK, query_count=1)
    page = make_page()
    adapter = FakeResearchAdapter(
        search_results=[WebSearchResult(url=page.final_url, title=page.title)],
        page=page,
    )
    request = ResearchRequest(
        plan=plan,
        task=ResearchTask.for_workstream(plan, "market_analysis"),
        repair_task=EvidenceRepairTask(
            task_id="repair_measured_review",
            objective="Capture an independent measured review.",
            research_question_ids=["market_size"],
            section_ids=["market_findings"],
            candidate_queries=["independent measured EV market review"],
        ),
        budget_usage=BudgetUsage(searches=5),
    )

    result = await ResearchAgent(FakeGateway(make_synthesis(page)), adapter).research(request)

    assert result.budget_usage.searches == 6
    assert adapter.web_calls[0].query == "independent measured EV market review"


def test_unreferenced_material_is_not_an_accepted_source() -> None:
    used_page = make_page()
    unused_page = used_page.model_copy(
        update={
            "final_url": "https://example.com/unused",
            "title": "Unused commentary",
            "content": used_page.content.replace("20 percent", "18 percent"),
        }
    )
    materials = [
        ResearchAgent._material_from_page(used_page),
        ResearchAgent._material_from_page(unused_page),
    ]
    segments = _segment_materials(materials)

    evidence = ResearchAgent._finalize_evidence(
        make_synthesis(used_page), materials, segments
    )

    assert len(evidence.sources) == 1
    assert evidence.sources[0].canonical_url == "https://example.gov/ev"


def test_quick_claim_limits_bound_initial_and_repair_synthesis() -> None:
    plan = make_plan(preset=DepthPreset.QUICK, query_count=1)
    page = make_page()
    material = ResearchAgent._material_from_page(page)
    segments = _segment_materials([material])
    base_claim = make_synthesis(page).claims[0]
    initial_request = ResearchRequest(
        plan=plan,
        task=ResearchTask.for_workstream(plan, "market_analysis"),
    )
    eight_claims = ResearchSynthesisDraft(
        claims=[
            base_claim.model_copy(
                update={"normalized_claim": f"Initial finding number {index}."}
            )
            for index in range(8)
        ]
    )

    with pytest.raises(ValueError, match="7-claim ceiling"):
        ResearchAgent._validate_synthesis(
            eight_claims, initial_request, [material], segments
        )

    repair_request = initial_request.model_copy(
        update={
            "repair_task": EvidenceRepairTask(
                task_id="repair_measurement",
                objective="Confirm a missing measurement.",
                research_question_ids=["market_size"],
                section_ids=["market_findings"],
                candidate_queries=["measured EV market result"],
            )
        }
    )
    four_claims = ResearchSynthesisDraft(
        claims=[
            base_claim.model_copy(
                update={"normalized_claim": f"Repair finding number {index}."}
            )
            for index in range(4)
        ]
    )

    with pytest.raises(ValueError, match="3-claim ceiling"):
        ResearchAgent._validate_synthesis(
            four_claims, repair_request, [material], segments
        )


@pytest.mark.asyncio
async def test_upload_search_uses_run_bound_adapter_and_preserves_location() -> None:
    request = make_request(uses_uploads=True)
    operation_plan = ResearchExecutionPlan(
        upload_searches=[
            UploadSearchOperation(
                query="internal EV forecast",
                research_question_ids=["market_size"],
            )
        ],
        rationale="Use the approved private-document scope.",
    )
    chunk = UploadChunk(
        upload_id="upload-123",
        filename="forecast.pdf",
        title="Internal EV forecast",
        content="The internal forecast estimates 2 million EV sales in 2027.",
        location="Page 4, paragraph 3",
        document_hash="b" * 64,
        access_date="2026-08-27",
    )
    material = ResearchAgent._material_from_upload(chunk)
    segment = _segment_materials([material])[0]
    synthesis = ResearchSynthesisDraft(
        claims=[
            DraftResearchClaim(
                research_question_id="market_size",
                section_ids=["market_findings"],
                normalized_claim="The internal forecast estimates 2 million sales in 2027.",
                evidence=[
                    DraftEvidenceSelection(
                        material_id=material.material_id,
                        segment_id=segment.segment_id,
                    )
                ],
                support_strength=SupportStrength.MODERATE,
            )
        ]
    )
    adapter = FakeResearchAdapter(upload_chunks=[chunk])

    result = await ResearchAgent(
        FakeGateway(operation_plan, synthesis), adapter
    ).research(request)

    assert adapter.web_calls == []
    assert len(adapter.upload_calls) == 1
    assert result.evidence.sources[0].upload_name == "forecast.pdf"
    assert result.evidence.excerpts[0].location.startswith("Page 4, paragraph 3")


def test_canonicalize_url_removes_tracking_but_preserves_content_query() -> None:
    assert canonicalize_url(
        "https://EXAMPLE.com/report?id=7&utm_source=email#results"
    ) == "https://example.com/report?id=7"


def test_research_request_rejects_task_scope_tampering() -> None:
    request = make_request()
    tampered_task = request.task.model_copy(update={"objective": "Research an unrelated topic."})

    with pytest.raises(ValidationError, match="objective differs"):
        ResearchRequest(plan=request.plan, task=tampered_task)


@pytest.mark.asyncio
async def test_parallel_workstream_results_merge_by_stable_evidence_ids() -> None:
    page = make_page()
    adapter = FakeResearchAdapter(
        search_results=[WebSearchResult(url="https://example.gov/ev", title="EV")],
        page=page,
    )
    result = await ResearchAgent(
        FakeGateway(make_synthesis(page)), adapter
    ).research(make_request())

    merged = merge_research_results([result, result])

    assert len(merged.sources) == 1
    assert len(merged.excerpts) == 1
    assert len(merged.claims) == 1
