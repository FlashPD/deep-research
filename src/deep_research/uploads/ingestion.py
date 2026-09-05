import asyncio
import hashlib
import json
import os
import struct
import zipfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path, PurePath
from typing import Protocol

from deep_research.uploads.index import UploadIndex
from deep_research.uploads.models import (
    IndexedUploadChunk,
    ParsedBlock,
    UploadIngestionResult,
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_EXTRACTED_CHARS = 1_000_000
MAX_PDF_PAGES = 500
MAX_DOCX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
_MEDIA_TYPES = {
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    },
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/plain"},
}


class UploadValidationError(ValueError):
    pass


class MalwareDetectedError(ValueError):
    pass


class MalwareScanner(Protocol):
    async def scan(self, content: bytes) -> None: ...


class UploadArtifactStore(Protocol):
    async def quarantine(self, upload_id: str, filename: str, content: bytes) -> str: ...

    async def promote(self, upload_id: str, filename: str) -> str: ...

    async def store_extracted(
        self, upload_id: str, chunks: list[IndexedUploadChunk]
    ) -> str: ...

    async def delete_quarantine(self, upload_id: str, filename: str) -> None: ...


class ClamAVScanner:
    """Streams bytes to clamd without exposing a filesystem path to the scanner."""

    def __init__(self, host: str = "localhost", port: int = 3310, timeout: float = 30) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout

    async def scan(self, content: bytes) -> None:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port), timeout=self._timeout
        )
        try:
            writer.write(b"zINSTREAM\0")
            for offset in range(0, len(content), 64 * 1024):
                chunk = content[offset : offset + 64 * 1024]
                writer.write(struct.pack("!I", len(chunk)))
                writer.write(chunk)
            writer.write(struct.pack("!I", 0))
            await writer.drain()
            response = await asyncio.wait_for(reader.read(4096), timeout=self._timeout)
        finally:
            writer.close()
            await writer.wait_closed()
        result = response.decode("utf-8", errors="replace").strip("\x00\r\n ")
        if result.endswith("FOUND"):
            raise MalwareDetectedError("malware scanner rejected the upload")
        if not result.endswith("OK"):
            raise RuntimeError("malware scanner returned an indeterminate result")


class LocalUploadArtifactStore:
    """Private local-development storage mirroring quarantine/original/extracted prefixes."""

    def __init__(self, root: Path, *, tenant_id: str, run_id: str) -> None:
        if not tenant_id or not run_id:
            raise ValueError("tenant_id and run_id are required")
        scope = Path(_scope_key(tenant_id)) / _scope_key(run_id)
        self._root = root.resolve() / scope

    async def quarantine(self, upload_id: str, filename: str, content: bytes) -> str:
        safe_name = PurePath(filename).name
        path = self._root / "quarantine" / _scope_key(upload_id) / safe_name
        await asyncio.to_thread(_private_write, path, content)
        return str(path)

    async def promote(self, upload_id: str, filename: str) -> str:
        safe_name = PurePath(filename).name
        source = self._root / "quarantine" / _scope_key(upload_id) / safe_name
        target = self._root / "originals" / _scope_key(upload_id) / safe_name
        await asyncio.to_thread(_private_move, source, target)
        return str(target)

    async def store_extracted(
        self, upload_id: str, chunks: list[IndexedUploadChunk]
    ) -> str:
        path = self._root / "extracted" / f"{_scope_key(upload_id)}.json"
        content = json.dumps(
            [chunk.model_dump(mode="json") for chunk in chunks],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        await asyncio.to_thread(_private_write, path, content)
        return str(path)

    async def delete_quarantine(self, upload_id: str, filename: str) -> None:
        safe_name = PurePath(filename).name
        path = self._root / "quarantine" / _scope_key(upload_id) / safe_name
        await asyncio.to_thread(path.unlink, missing_ok=True)


class UploadIngestionService:
    def __init__(
        self,
        *,
        index: UploadIndex,
        artifact_store: UploadArtifactStore,
        malware_scanner: MalwareScanner,
        chunk_chars: int = 4_000,
        overlap_chars: int = 400,
    ) -> None:
        if chunk_chars < 500 or overlap_chars < 0 or overlap_chars >= chunk_chars:
            raise ValueError("invalid upload chunking configuration")
        self._index = index
        self._artifact_store = artifact_store
        self._malware_scanner = malware_scanner
        self._chunk_chars = chunk_chars
        self._overlap_chars = overlap_chars

    async def ingest(
        self,
        *,
        upload_id: str,
        filename: str,
        declared_media_type: str,
        content: bytes,
        access_date: str,
    ) -> UploadIngestionResult:
        if not upload_id:
            raise UploadValidationError("upload_id is required")
        safe_filename = PurePath(filename).name
        if safe_filename != filename or not safe_filename or "\\" in filename:
            raise UploadValidationError("filename must not contain a path")
        await self._artifact_store.quarantine(upload_id, safe_filename, content)
        try:
            media_type = _validate_upload(safe_filename, declared_media_type, content)
            await self._malware_scanner.scan(content)
            document_hash = hashlib.sha256(content).hexdigest()
            blocks = await asyncio.to_thread(_parse_document, safe_filename, content)
            if sum(len(block.text) for block in blocks) > MAX_EXTRACTED_CHARS:
                raise UploadValidationError("extracted document text exceeds the limit")
            chunks = _chunk_blocks(
                upload_id=upload_id,
                filename=safe_filename,
                document_hash=document_hash,
                access_date=access_date,
                blocks=blocks,
                chunk_chars=self._chunk_chars,
                overlap_chars=self._overlap_chars,
            )
            if not chunks:
                raise UploadValidationError("document contains no extractable text")
            extracted_location = await self._artifact_store.store_extracted(upload_id, chunks)
            original_location = await self._artifact_store.promote(upload_id, safe_filename)
            await self._index.delete_upload(upload_id)
            await self._index.index_chunks(chunks)
            return UploadIngestionResult(
                upload_id=upload_id,
                filename=safe_filename,
                media_type=media_type,
                size_bytes=len(content),
                document_hash=document_hash,
                block_count=len(blocks),
                chunk_count=len(chunks),
                original_location=original_location,
                extracted_location=extracted_location,
            )
        except BaseException:
            await asyncio.shield(
                self._artifact_store.delete_quarantine(upload_id, safe_filename)
            )
            raise


def _validate_upload(filename: str, declared_media_type: str, content: bytes) -> str:
    if not content:
        raise UploadValidationError("upload is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadValidationError("upload exceeds the 25 MB file limit")
    extension = Path(filename).suffix.casefold()
    if extension not in _MEDIA_TYPES:
        raise UploadValidationError("unsupported upload extension")
    media_type = declared_media_type.split(";", 1)[0].strip().casefold()
    if media_type not in _MEDIA_TYPES[extension]:
        raise UploadValidationError("declared media type does not match the extension")
    if extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise UploadValidationError("PDF magic bytes are invalid")
    if extension == ".docx":
        _validate_docx_archive(content)
    if extension in {".txt", ".md"}:
        if b"\x00" in content:
            raise UploadValidationError("text upload contains binary null bytes")
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UploadValidationError("text upload must be UTF-8") from exc
    return media_type


def _validate_docx_archive(content: bytes) -> None:
    if not content.startswith(b"PK"):
        raise UploadValidationError("DOCX magic bytes are invalid")
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            names = set(archive.namelist())
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise UploadValidationError("DOCX archive structure is invalid")
            uncompressed = sum(item.file_size for item in archive.infolist())
            if uncompressed > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise UploadValidationError("DOCX expanded size exceeds the limit")
    except zipfile.BadZipFile as exc:
        raise UploadValidationError("DOCX archive is invalid") from exc


def _parse_document(filename: str, content: bytes) -> list[ParsedBlock]:
    extension = Path(filename).suffix.casefold()
    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise UploadValidationError("encrypted PDFs are not supported")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise UploadValidationError("PDF page count exceeds the limit")
        return [
            ParsedBlock(text=text, location=f"Page {number}", ordinal=number - 1)
            for number, page in enumerate(reader.pages, start=1)
            if (text := (page.extract_text() or "").strip())
        ]
    if extension == ".docx":
        from docx import Document

        document = Document(BytesIO(content))
        blocks = []
        for ordinal, paragraph in enumerate(document.paragraphs):
            text = paragraph.text.strip()
            if not text:
                continue
            style = paragraph.style.name if paragraph.style else "Paragraph"
            blocks.append(
                ParsedBlock(
                    text=text,
                    location=f"{style}, paragraph {ordinal + 1}",
                    ordinal=ordinal,
                )
            )
        return blocks
    return _parse_text_blocks(content.decode("utf-8-sig"))


def _parse_text_blocks(text: str) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    current: list[str] = []
    start_line = 1
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            if not current:
                start_line = line_number
            current.append(line)
        elif current:
            blocks.append(
                ParsedBlock(
                    text="\n".join(current).strip(),
                    location=f"Lines {start_line}-{line_number - 1}",
                    ordinal=len(blocks),
                )
            )
            current = []
    if current:
        blocks.append(
            ParsedBlock(
                text="\n".join(current).strip(),
                location=f"Lines {start_line}-{len(text.splitlines())}",
                ordinal=len(blocks),
            )
        )
    return blocks


def _chunk_blocks(
    *,
    upload_id: str,
    filename: str,
    document_hash: str,
    access_date: str,
    blocks: Sequence[ParsedBlock],
    chunk_chars: int,
    overlap_chars: int,
) -> list[IndexedUploadChunk]:
    chunks: list[IndexedUploadChunk] = []
    for block in blocks:
        start = 0
        while start < len(block.text):
            end = min(start + chunk_chars, len(block.text))
            text = block.text[start:end]
            location = block.location
            if len(block.text) > chunk_chars:
                location = f"{location}, characters {start}-{end - 1}"
            chunk_id = hashlib.sha256(
                f"{upload_id}\x00{document_hash}\x00{location}\x00{text}".encode()
            ).hexdigest()
            chunks.append(
                IndexedUploadChunk(
                    chunk_id=chunk_id,
                    upload_id=upload_id,
                    filename=filename,
                    title=filename,
                    content=text,
                    location=location,
                    document_hash=document_hash,
                    access_date=access_date,
                    ordinal=len(chunks),
                )
            )
            if end == len(block.text):
                break
            start = end - overlap_chars
    return chunks


def _scope_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _private_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _private_move(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.replace(source, target)
    target.chmod(0o600)
