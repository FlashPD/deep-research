# Deep Research

Python 3.13 backend foundation for the architecture in
[`arch_plan/deep-research-system-plan.md`](arch_plan/deep-research-system-plan.md).

Implemented so far:

- typed clarification and normalized research-brief contracts;
- a three-round, no-tools Clarifier Agent with explicit final confirmation;
- a no-tools Planning Agent with validated workstream DAGs and fixed depth ceilings;
- a bounded Research Agent with approved workstream tasks, typed operation planning, run-bound
  adapter interfaces, fetched-source enforcement, exact-excerpt validation, and stable evidence IDs;
- deterministic plan versions and canonical SHA-256 approval hashes;
- a no-tools Questions Agent producing report-linked, prioritized follow-up topics;
- a bounded Evidence Reviewer with source scoring, deterministic coverage checks, targeted repair
  tasks, budget-aware retry ceilings, and explicit exhausted-budget limitations;
- a Report Generation Agent with evidence-bound reviews, claim-mapped citation validation,
  deterministic Markdown assembly and source appendices, checksums, and strict Mermaid degradation;
- configuration-driven Bedrock, Anthropic, and OpenAI Strands model routing;
- whole-operation retries and provider fallback without mixing partial outputs;
- FastAPI endpoints for each implemented agent plus a health check;
- credential-free unit tests using fake structured model responses.

## Local setup

The checked-in package requires Python 3.13. A local `.venv` has been created with Python 3.13.15.

```bash
source .venv/bin/activate
pip install -e '.[dev]'
playwright install chromium
cp .env.example .env
pytest
uvicorn deep_research.api.app:app --reload
```

Model routes live in [`config/models.yaml`](config/models.yaml). Bedrock uses the normal AWS
credential chain. Direct Anthropic and OpenAI fallbacks become eligible only when their respective
API-key environment variables are present.

## Agent endpoints

```text
POST /v1/clarifier/evaluate
POST /v1/planner/generate
POST /v1/researcher/research
POST /v1/questions/generate
POST /v1/reviewer/review
POST /v1/report/generate
```

The request carries the topic, safe upload metadata, previous answers, current normalized brief,
and clarification round. The response is always a validated `ClarificationDecision`; the endpoint
does not expose public-web or retrieval tools.

The planning endpoint accepts an approved brief, depth preset, upload metadata, edits, and an
optional previous plan. Budget ceilings, version, and content hash are application-controlled. The
questions endpoint accepts a compact completed-report context and returns five to ten suggestions;
it never starts a new run.

The reviewer result is bound to hashes of the exact plan and evidence package. Repair tasks cannot
exceed the plan's remaining query or retry budget. The report endpoint accepts only a non-repair
review for those same inputs, rejects invented or unmapped citations, and omits unsafe Mermaid after
one repair attempt while retaining a prose fallback.

## Research adapter boundary

`ResearchAgent` never receives credentials, tenant IDs, index clients, or browser sessions from the
model. A run-bound `ResearchAdapter` supplies three typed capabilities: `search_web`, `fetch_page`,
and `search_uploads`. The default API application adapter remains deliberately unconfigured because
the API foundation does not yet authenticate and resolve a run owner. A durable worker should call
`build_live_research_services(settings, tenant_id=..., run_id=...)` only after ownership and exact
plan approval have been verified.

The live bundle uses Tavily for discovery, an isolated Playwright Chromium context for opened-page
evidence, and a tenant/run-filtered OpenSearch index for upload chunks. Tavily snippets remain
discovery metadata. The Playwright adapter revalidates public HTTP(S) destinations, blocks local and
private network requests, limits redirects and response sizes, and extracts rendered HTML or public
PDF text. OpenSearch supports local basic authentication and AWS SigV4 for Serverless (`aoss`).

Search results are discovery metadata only. Public evidence must pass through `fetch_page`, while
upload search returns bounded chunks with preserved document locations. Parallel workstream results
are combined with `merge_research_results` before constructing a `ReviewerRequest`.

## Upload ingestion and retrieval

`UploadIngestionService` is the post-upload processing boundary. The browser-facing presigned-upload
API is intentionally not exposed yet because authentication, run persistence, S3, and quarantine
events have not been implemented. Once an authenticated API or worker has the bytes, ingestion is:

1. Write the original into a tenant/run-scoped quarantine location.
2. Enforce the 25 MB limit and validate extension, declared MIME type, and magic/archive structure.
3. Stream the bytes to ClamAV. Scanner failure or an indeterminate result fails closed.
4. Parse PDF pages, DOCX paragraphs/headings, or UTF-8 TXT/Markdown line blocks.
5. Create overlapping, location-preserving chunks with stable content hashes.
6. Store the private original and extracted chunk artifact separately with local `0600` permissions.
7. Replace that upload's index entries and write chunks to OpenSearch with server-injected
   `tenant_id` and `run_id` fields.
8. At research time, inject those same identity filters into every query and return only bounded
   `UploadChunk` excerpts and coordinates to the Research Agent.

The local artifact store mirrors the planned quarantine/original/extracted prefixes. Production S3
storage and presigned upload endpoints remain part of the persistence/authentication workstream; the
parsing and index contracts do not need to change when that store is added.
