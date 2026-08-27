import hashlib

import pytest
from pydantic import ValidationError

from deep_research.agents.report import ReportGenerationAgent, ReportValidationError
from deep_research.agents.reviewer import EvidenceReviewer
from deep_research.contracts.evidence import ReviewerRequest, ReviewState
from deep_research.contracts.reporting import MermaidDiagram, ReportRequest
from tests.conftest import FakeGateway
from tests.factories import (
    make_evidence_package,
    make_plan,
    make_report_draft,
    make_review_draft,
)


async def _approved_review():
    return await EvidenceReviewer(FakeGateway(make_review_draft())).review(
        ReviewerRequest(plan=make_plan(), evidence=make_evidence_package())
    )


@pytest.mark.asyncio
async def test_report_agent_builds_canonical_cited_markdown() -> None:
    gateway = FakeGateway(make_report_draft())
    request = ReportRequest(
        plan=make_plan(), evidence=make_evidence_package(), review=await _approved_review()
    )

    artifact = await ReportGenerationAgent(gateway).generate(request)

    assert "## Sources" in artifact.markdown
    assert "https://example.gov/ev-statistics" in artifact.markdown
    assert artifact.cited_source_ids == ["S1"]
    assert artifact.checksum == hashlib.sha256(artifact.markdown.encode()).hexdigest()
    assert gateway.calls[0]["role"] == "report"


@pytest.mark.asyncio
async def test_report_agent_repairs_an_unmapped_citation() -> None:
    gateway = FakeGateway(
        make_report_draft(content="The market doubled. [S2]"),
        make_report_draft(),
    )
    request = ReportRequest(
        plan=make_plan(), evidence=make_evidence_package(), review=await _approved_review()
    )

    artifact = await ReportGenerationAgent(gateway).generate(request)

    assert len(gateway.calls) == 2
    assert "unknown or malformed" in gateway.calls[1]["prompt"]
    assert artifact.cited_source_ids == ["S1"]


@pytest.mark.asyncio
async def test_invalid_mermaid_is_omitted_after_one_repair() -> None:
    invalid = make_report_draft().model_copy(
        update={
            "diagrams": [
                MermaidDiagram(
                    section_id="market_findings",
                    title="Unsafe diagram",
                    code='flowchart TD\nA-->B\nclick A "https://example.com"',
                    fallback_text="The market progresses from measurement to interpretation.",
                )
            ]
        }
    )
    gateway = FakeGateway(invalid, invalid)
    request = ReportRequest(
        plan=make_plan(), evidence=make_evidence_package(), review=await _approved_review()
    )

    artifact = await ReportGenerationAgent(gateway).generate(request)

    assert "```mermaid" not in artifact.markdown
    assert "Diagram omitted under strict validation" in artifact.markdown
    assert artifact.mermaid_validation[0].valid is False


@pytest.mark.asyncio
async def test_valid_flowchart_survives_strict_mermaid_validation() -> None:
    draft = make_report_draft().model_copy(
        update={
            "diagrams": [
                MermaidDiagram(
                    section_id="market_findings",
                    title="Evidence flow",
                    code="flowchart TD\nA[Evidence] --> B[Finding]",
                    fallback_text="Evidence supports the finding.",
                )
            ]
        }
    )
    request = ReportRequest(
        plan=make_plan(), evidence=make_evidence_package(), review=await _approved_review()
    )

    artifact = await ReportGenerationAgent(FakeGateway(draft)).generate(request)

    assert "```mermaid" in artifact.markdown
    assert artifact.mermaid_validation[0].valid is True


@pytest.mark.asyncio
async def test_report_generation_blocks_while_repair_is_required() -> None:
    review = await _approved_review()
    review = review.model_copy(update={"review_state": ReviewState.REPAIR_REQUIRED})

    with pytest.raises(ValueError, match="repair is required"):
        await ReportGenerationAgent(FakeGateway()).generate(
            ReportRequest(plan=make_plan(), evidence=make_evidence_package(), review=review)
        )


@pytest.mark.asyncio
async def test_report_request_rejects_review_for_different_evidence() -> None:
    review = await _approved_review()
    changed_evidence = make_evidence_package().model_copy(update={"claims": []})

    with pytest.raises(ValidationError, match="different evidence package"):
        ReportRequest(plan=make_plan(), evidence=changed_evidence, review=review)


@pytest.mark.asyncio
async def test_report_remains_blocked_after_bad_citation_repair() -> None:
    bad = make_report_draft(content="The market doubled without a citation.")
    request = ReportRequest(
        plan=make_plan(), evidence=make_evidence_package(), review=await _approved_review()
    )

    with pytest.raises(ReportValidationError):
        await ReportGenerationAgent(FakeGateway(bad, bad)).generate(request)
