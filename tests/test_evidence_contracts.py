import pytest
from pydantic import ValidationError

from deep_research.contracts.evidence import EvidencePackage, SourceRecord, SourceType
from tests.factories import make_evidence_package


def test_evidence_package_rejects_unresolved_excerpt_source() -> None:
    payload = make_evidence_package().model_dump(mode="json")
    payload["excerpts"][0]["source_id"] = "S2"

    with pytest.raises(ValidationError, match="unknown sources"):
        EvidencePackage.model_validate(payload)


def test_upload_source_cannot_smuggle_a_public_url() -> None:
    with pytest.raises(ValidationError, match="cannot contain a URL"):
        SourceRecord(
            source_id="S1",
            source_type=SourceType.UPLOAD,
            title="Private memo",
            access_date="2026-08-26",
            canonical_url="https://example.com/private",
            upload_name="memo.pdf",
            content_hash="a" * 64,
        )
