import hashlib

import pytest
from pydantic import ValidationError

from deep_research.agents.report import ReportGenerationAgent, ReportValidationError
from deep_research.agents.reviewer import EvidenceReviewer
from deep_research.contracts.evidence import ReviewerRequest, ReviewState
from deep_research.contracts.planning import DepthPreset
from deep_research.contracts.reporting import MermaidDiagram, ReportRequest
from deep_research.models.config import ModelProvider
from deep_research.models.gateway import ModelAttemptFailure, ModelInvocationError
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
    assert "Compact report input" in gateway.calls[0]["prompt"]
    assert "claims_by_section" in gateway.calls[0]["prompt"]
    assert "question_coverage" not in gateway.calls[0]["prompt"]


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

    with pytest.raises(ValidationError, match="accepted research evidence"):
        ReportRequest(plan=make_plan(), evidence=make_evidence_package(), review=review)


@pytest.mark.asyncio
async def test_report_request_rejects_empty_evidence() -> None:
    evidence = make_evidence_package().model_copy(
        update={"sources": [], "excerpts": [], "claims": []}
    )
    review = (await _approved_review()).model_copy(
        update={"evidence_checksum": evidence.calculate_checksum()}
    )

    with pytest.raises(ValidationError, match="evidence-backed claim"):
        ReportRequest(plan=make_plan(), evidence=evidence, review=review)


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


def _output_limit_error() -> ModelInvocationError:
    return ModelInvocationError(
        "report",
        [
            ModelAttemptFailure(
                target_name="anthropic_report",
                provider=ModelProvider.ANTHROPIC,
                model_id="report-model",
                attempt=1,
                max_attempts=2,
                error_type="MaxTokensReachedException",
                message="Model stopped generating due to maximum token limit.",
            )
        ],
    )


@pytest.mark.asyncio
async def test_report_retries_once_with_compact_prompt_after_output_limit() -> None:
    gateway = FakeGateway(_output_limit_error(), make_report_draft())
    plan = make_plan(preset=DepthPreset.QUICK)
    evidence = make_evidence_package()
    review = await EvidenceReviewer(FakeGateway(make_review_draft())).review(
        ReviewerRequest(plan=plan, evidence=evidence)
    )
    request = ReportRequest(
        plan=plan,
        evidence=evidence,
        review=review,
    )

    artifact = await ReportGenerationAgent(gateway).generate(request)

    assert len(gateway.calls) == 2
    assert '"mode": "compact_retry"' in gateway.calls[1]["prompt"]
    assert artifact.generation_metadata.prompt_version == "report-v2"


@pytest.mark.asyncio
async def test_second_output_limit_uses_deterministic_cited_report() -> None:
    gateway = FakeGateway(_output_limit_error(), _output_limit_error())
    request = ReportRequest(
        plan=make_plan(), evidence=make_evidence_package(), review=await _approved_review()
    )

    artifact = await ReportGenerationAgent(gateway).generate(request)

    assert len(gateway.calls) == 2
    assert "deterministically from reviewed" in artifact.markdown
    assert "[S1]" in artifact.markdown
    assert artifact.generation_metadata.prompt_version.endswith("deterministic-fallback")


@pytest.mark.asyncio
async def test_quick_report_rejects_oversized_sections() -> None:
    plan = make_plan(preset=DepthPreset.QUICK)
    evidence = make_evidence_package()
    review = await EvidenceReviewer(FakeGateway(make_review_draft())).review(
        ReviewerRequest(plan=plan, evidence=evidence)
    )
    oversized = make_report_draft(
        content=("Measured market growth remained positive. [S1]\n\n" * 50)
    )
    request = ReportRequest(plan=plan, evidence=evidence, review=review)

    with pytest.raises(ReportValidationError, match="exceeds 1800 characters"):
        await ReportGenerationAgent(FakeGateway(oversized, oversized)).generate(request)
