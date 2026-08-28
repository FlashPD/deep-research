import asyncio
import re
from collections import Counter
from typing import Any, Protocol
from urllib.parse import urlsplit

from deep_research.contracts.research import UploadChunk, UploadSearchOperation
from deep_research.uploads.models import IndexedUploadChunk

_INDEX_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,254}$")
_TOKENS = re.compile(r"[a-z0-9]{2,}")


class UploadIndex(Protocol):
    """Tenant/run-bound index; identity is never accepted from model-authored operations."""

    async def index_chunks(self, chunks: list[IndexedUploadChunk]) -> None: ...

    async def search(self, query: str, max_chunks: int) -> list[IndexedUploadChunk]: ...

    async def delete_upload(self, upload_id: str) -> None: ...


class UploadIndexSearchAdapter:
    def __init__(self, index: UploadIndex) -> None:
        self._index = index

    async def search_uploads(self, operation: UploadSearchOperation) -> list[UploadChunk]:
        results = await self._index.search(operation.query, operation.max_chunks)
        return [
            UploadChunk(
                upload_id=item.upload_id,
                filename=item.filename,
                title=item.title,
                content=item.content,
                location=item.location,
                document_hash=item.document_hash,
                access_date=item.access_date,
            )
            for item in results
        ]


class InMemoryUploadIndex:
    """Run-bound local/test implementation with deterministic lexical ranking."""

    def __init__(self) -> None:
        self._chunks: dict[str, IndexedUploadChunk] = {}
        self._lock = asyncio.Lock()

    async def index_chunks(self, chunks: list[IndexedUploadChunk]) -> None:
        async with self._lock:
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk

    async def search(self, query: str, max_chunks: int) -> list[IndexedUploadChunk]:
        query_terms = Counter(_TOKENS.findall(query.casefold()))

        def score(chunk: IndexedUploadChunk) -> tuple[int, int]:
            terms = Counter(_TOKENS.findall(f"{chunk.title} {chunk.content}".casefold()))
            lexical_score = sum(
                min(count, terms.get(term, 0)) for term, count in query_terms.items()
            )
            return lexical_score, -chunk.ordinal

        ranked = sorted(self._chunks.values(), key=score, reverse=True)
        return [chunk for chunk in ranked if score(chunk)[0] > 0][:max_chunks]

    async def delete_upload(self, upload_id: str) -> None:
        async with self._lock:
            self._chunks = {
                chunk_id: chunk
                for chunk_id, chunk in self._chunks.items()
                if chunk.upload_id != upload_id
            }


class OpenSearchUploadIndex:
    """OpenSearch/OpenSearch Serverless implementation with mandatory identity filters."""

    def __init__(
        self,
        client: Any,
        *,
        tenant_id: str,
        run_id: str,
        index_name: str = "deep-research-upload-chunks",
    ) -> None:
        if not tenant_id or not run_id:
            raise ValueError("tenant_id and run_id are required")
        if not _INDEX_NAME.fullmatch(index_name):
            raise ValueError("invalid OpenSearch index name")
        self._client = client
        self._tenant_id = tenant_id
        self._run_id = run_id
        self._index_name = index_name
        self._ready = False
        self._ready_lock = asyncio.Lock()

    @classmethod
    def from_url(
        cls,
        *,
        url: str,
        tenant_id: str,
        run_id: str,
        index_name: str = "deep-research-upload-chunks",
        username: str | None = None,
        password: str | None = None,
        aws_region: str | None = None,
        aws_service: str = "aoss",
    ) -> "OpenSearchUploadIndex":
        from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OpenSearch URL must be absolute HTTP(S)")
        auth: Any = None
        if aws_region:
            import boto3

            credentials = boto3.Session().get_credentials()
            if credentials is None:
                raise RuntimeError("AWS credentials are unavailable for OpenSearch")
            auth = AWSV4SignerAuth(credentials, aws_region, aws_service)
        elif username:
            auth = (username, password or "")
        client = OpenSearch(
            hosts=[
                {
                    "host": parsed.hostname,
                    "port": parsed.port or (443 if parsed.scheme == "https" else 9200),
                }
            ],
            http_auth=auth,
            use_ssl=parsed.scheme == "https",
            verify_certs=parsed.scheme == "https",
            connection_class=RequestsHttpConnection,
            pool_maxsize=20,
        )
        return cls(
            client,
            tenant_id=tenant_id,
            run_id=run_id,
            index_name=index_name,
        )

    async def index_chunks(self, chunks: list[IndexedUploadChunk]) -> None:
        await self._ensure_index()
        body: list[dict[str, Any]] = []
        for chunk in chunks:
            body.extend(
                [
                    {
                        "index": {
                            "_index": self._index_name,
                            "_id": f"{self._tenant_id}:{self._run_id}:{chunk.chunk_id}",
                        }
                    },
                    {
                        **chunk.model_dump(mode="json"),
                        "tenant_id": self._tenant_id,
                        "run_id": self._run_id,
                    },
                ]
            )
        if not body:
            return
        response = await asyncio.to_thread(self._client.bulk, body=body, refresh=True)
        if response.get("errors"):
            raise RuntimeError("OpenSearch rejected one or more upload chunks")

    async def search(self, query: str, max_chunks: int) -> list[IndexedUploadChunk]:
        await self._ensure_index()
        response = await asyncio.to_thread(
            self._client.search,
            index=self._index_name,
            body={
                "size": max_chunks,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"tenant_id": self._tenant_id}},
                            {"term": {"run_id": self._run_id}},
                        ],
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content", "title^2"],
                                }
                            }
                        ],
                    }
                },
            },
        )
        return [
            IndexedUploadChunk.model_validate(
                {
                    key: value
                    for key, value in hit["_source"].items()
                    if key not in {"tenant_id", "run_id"}
                }
            )
            for hit in response.get("hits", {}).get("hits", [])
        ]

    async def delete_upload(self, upload_id: str) -> None:
        await self._ensure_index()
        await asyncio.to_thread(
            self._client.delete_by_query,
            index=self._index_name,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"tenant_id": self._tenant_id}},
                            {"term": {"run_id": self._run_id}},
                            {"term": {"upload_id": upload_id}},
                        ]
                    }
                }
            },
            refresh=True,
        )

    async def _ensure_index(self) -> None:
        if self._ready:
            return
        async with self._ready_lock:
            if self._ready:
                return
            exists = await asyncio.to_thread(
                self._client.indices.exists, index=self._index_name
            )
            if not exists:
                try:
                    await asyncio.to_thread(
                        self._client.indices.create,
                        index=self._index_name,
                        body={"mappings": {"properties": _INDEX_MAPPING}},
                    )
                except Exception as exc:
                    if "resource_already_exists_exception" not in str(exc):
                        raise
            self._ready = True


_INDEX_MAPPING: dict[str, Any] = {
    "tenant_id": {"type": "keyword"},
    "run_id": {"type": "keyword"},
    "chunk_id": {"type": "keyword"},
    "upload_id": {"type": "keyword"},
    "filename": {"type": "keyword"},
    "title": {"type": "text"},
    "content": {"type": "text"},
    "location": {"type": "keyword", "ignore_above": 4096},
    "document_hash": {"type": "keyword"},
    "access_date": {"type": "date"},
    "ordinal": {"type": "integer"},
}
