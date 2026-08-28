from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

ShortText = Annotated[str, Field(min_length=1, max_length=4_000)]


class ParsedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Annotated[str, Field(min_length=1, max_length=100_000)]
    location: ShortText
    ordinal: int = Field(ge=0)


class IndexedUploadChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    upload_id: Annotated[str, Field(min_length=1, max_length=500)]
    filename: Annotated[str, Field(min_length=1, max_length=500)]
    title: ShortText
    content: Annotated[str, Field(min_length=1, max_length=50_000)]
    location: ShortText
    document_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    access_date: Annotated[str, Field(min_length=1, max_length=100)]
    ordinal: int = Field(ge=0)


class UploadIngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: Annotated[str, Field(min_length=1, max_length=500)]
    filename: Annotated[str, Field(min_length=1, max_length=500)]
    media_type: Annotated[str, Field(min_length=1, max_length=200)]
    size_bytes: int = Field(ge=0)
    document_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    block_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    original_location: ShortText
    extracted_location: ShortText
