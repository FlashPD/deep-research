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
- an authenticated run control plane with validated lifecycle transitions, exact plan approval,
  idempotent commands, optimistic concurrency, cancellation, budgets, and replayable events;
- tenant/owner-aware in-memory and DynamoDB repositories with 30-day TTL metadata;
- a versioned phase-job protocol with acknowledged in-memory and SQS dispatch adapters;
- a durable worker with optimistic, idempotent graph checkpoints, bounded parallel workstreams,
  cooperative cancellation, monotonic usage accounting, and deterministic finalization;
- Cognito access-token verification with cached JWKS, issuer, signature, app-client, token-use,
  expiry, identity, and scope checks;
- FastAPI command endpoints that persist and dispatch work, plus opt-in agent debug endpoints;
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
deep-research-worker
```

Local setup defaults to `AUTH_MODE=development`, which injects a fixed local identity and must not
be used in a shared or production deployment. Set `AUTH_MODE=cognito`, `COGNITO_ISSUER`, and
`COGNITO_CLIENT_ID` to require Cognito bearer access tokens. The API always derives owner and tenant
identity from the authenticated principal; run requests cannot supply either value.

Model routes live in [`config/models.yaml`](config/models.yaml). Bedrock uses the normal AWS
credential chain. Direct Anthropic and OpenAI fallbacks become eligible only when their respective
API-key environment variables are present.

## Worker and dispatch

Starting a run, completing clarification, and approving a plan enqueue typed phase jobs. The
worker executes one bounded graph phase per delivery and checkpoints before the approval interrupt
and after every completed outer node or research workstream. A replacement worker resumes from the
canonical checkpoint; duplicate Standard-queue delivery is safe through optimistic revisions and
node-scoped idempotency keys.

Without `SQS_JOBS_QUEUE_URL`, dispatch uses an in-memory acknowledged queue. API and worker must be
composed with the same dispatcher and repository in one process for that mode. Separate processes
require SQS plus DynamoDB. Run a configured worker with `deep-research-worker`.

The production API does not construct or expose agents. Direct agent endpoints are available only
when `create_app` receives an explicit gateway (or `expose_agent_debug_routes=True`) for isolated
development tests.

## Agent debug endpoints

```text
POST /v1/clarifier/evaluate
POST /v1/planner/generate
POST /v1/researcher/research
POST /v1/questions/generate
POST /v1/reviewer/review
POST /v1/report/generate
```

## Run control plane

```text
POST /v1/runs
GET  /v1/runs/{run_id}
POST /v1/runs/{run_id}/start
POST /v1/runs/{run_id}/clarifications
PUT  /v1/runs/{run_id}/plan
POST /v1/runs/{run_id}/plan/approve
POST /v1/runs/{run_id}/cancel
GET  /v1/runs/{run_id}/events?after={cursor}
```

Every mutating request requires an `Idempotency-Key` header of 8–200 characters. Reusing a key for
the same command returns the original response; reusing it for different input returns `409`.
Plan approval requires the exact current plan version and canonical SHA-256 hash. Lifecycle updates
atomically persist the new run revision, one immutable ordered event, and the idempotency response.
Event polling accepts a cursor so clients can reconnect without losing progress. Worker events
contain only safe phase metadata, checksums, counts, and budget totals—not prompts or evidence text.

Without `DYNAMODB_RUNS_TABLE`, runs use process-local memory for development and tests. When the
table is configured, the repository uses one DynamoDB table with string partition/sort keys named
`PK` and `SK`, plus a numeric `expires_at` TTL attribute. Enable DynamoDB TTL on `expires_at`.
Conditional transactional writes enforce revision and event ordering. The application does not
create the table; infrastructure remains the CDK layer's responsibility.

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
research execution belongs in a durable worker, not the API request process. That worker should call
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
API is intentionally not exposed yet because the S3 upload authorization and quarantine-event
workflow are not implemented. Once an authenticated API or worker has the bytes, ingestion is:

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
