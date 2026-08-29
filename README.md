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
- a local CLI that performs clarification, approval, progress polling, cancellation, and report
  writing through the same control-plane and worker boundaries as the API;
- credential-free mocked tests plus explicitly selected, bounded live tests.

## Fresh-machine local setup

The package requires Python 3.13. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
playwright install chromium
cp .env.example .env
```

Configure one provider in `.env`:

```dotenv
# OpenAI
MODEL_PROVIDER=openai
OPENAI_API_KEY=replace-me
OPENAI_MODEL_ID=gpt-5.4
```

```dotenv
# Anthropic
MODEL_PROVIDER=anthropic
ANTHROPIC_API_KEY=replace-me
ANTHROPIC_MODEL_ID=claude-sonnet-4-6
```

```dotenv
# Amazon Bedrock; credentials still come from the standard AWS credential chain
MODEL_PROVIDER=bedrock
AWS_REGION=us-east-1
DEFAULT_MODEL_ID=global.anthropic.claude-sonnet-4-6
```

For Bedrock, configure the AWS CLI/profile or exported AWS credential variables before starting.
For every public-web provider, also set:

```dotenv
TAVILY_API_KEY=replace-me
UPLOADS_ENABLED=false
```

The [official OpenAI quickstart](https://developers.openai.com/api/docs/quickstart) recommends
keeping `OPENAI_API_KEY` in a protected environment variable; this project additionally supports
loading it from the uncommitted `.env` file and passes it directly to the server-side Strands
client. Never commit `.env`.

Start the API and worker together:

```bash
deep-research-dev
```

The server listens on `http://127.0.0.1:8000`. Startup fails immediately when the selected direct
provider key, Bedrock credential chain, or Tavily key is unavailable.

Local setup defaults to `AUTH_MODE=development`, which injects a fixed local identity and must not
be used in a shared or production deployment. Set `AUTH_MODE=cognito`, `COGNITO_ISSUER`, and
`COGNITO_CLIENT_ID` to require Cognito bearer access tokens. The API always derives owner and tenant
identity from the authenticated principal; run requests cannot supply either value.

Model routes live in [`config/models.yaml`](config/models.yaml). Set `MODEL_PROVIDER` to `openai`,
`anthropic`, or `bedrock`; OpenAI and Anthropic require their matching key, while Bedrock uses the
normal AWS credential chain. `MODEL_FALLBACK_ORDER` optionally contains a comma-separated provider
order. The selected provider is always first, so direct-provider development never probes Bedrock
unless Bedrock is explicitly listed as a fallback. Keys loaded by `pydantic-settings` from `.env`
are passed directly to the matching Strands client and are never logged.

`deep-research-dev` is the single-process development runtime. It shares one in-memory repository,
dispatcher, control service, gateway, and durable worker with FastAPI, and stops the background
worker during application shutdown. It requires `TAVILY_API_KEY` at startup. The production
`deep_research.api.app:app` and `deep-research-worker` composition roots remain separate.

## Local end-to-end CLI

The CLI owns an in-process local runtime but still creates commands through `RunControlService`,
dispatches typed phase jobs, and lets `DurableWorker` invoke agents and research adapters.

```bash
deep-research-run "Compare grid-scale battery technologies" --depth quick
deep-research-run "Research heat-pump adoption" --provider anthropic --output reports/
deep-research-run "Summarize current fusion milestones" --auto-approve-plan
```

Supported options are:

- `--depth quick|standard|deep`
- `--provider openai|anthropic|bedrock`, overriding `MODEL_PROVIDER` for that invocation
- `--output PATH`, where a `.md` path is used directly and a directory receives `<run_id>.md`
- `--auto-approve-plan`, which is the only way to bypass the interactive approval prompt

The default flow displays typed clarification questions, submits answers for the current round,
prints the generated plan, asks for approval of its exact version and SHA-256 hash, polls safe
progress events, writes the completed Markdown report, and displays limitations and follow-up
questions. Press Ctrl-C while work is running to persist a cooperative cancellation request.

## Tests

Mocked tests are deterministic and require no credentials:

```bash
pytest
```

Live tests are excluded by default. After configuring the selected provider, Tavily, and Chromium,
run them explicitly:

```bash
pytest -m live
```

The live suite uses Quick budgets and covers every model role, provider routing, Tavily followed by
Playwright, clarification resume, exact approval, cancellation, a complete public-web workflow, and
report citation resolution. Missing credentials cause clean skips.

## Expected workflow

1. Create and start a run. The worker invokes the tool-free Clarifier.
2. If questions are checkpointed, submit typed answers for that exact round. After three rounds,
   proceeding requires explicit final confirmation.
3. Review the generated, versioned plan. No Tavily, Playwright, or upload operation occurs before
   its exact version and hash are approved.
4. Poll cursor-based events while research, review, report generation, and follow-up generation run.
5. Read the completed report from the CLI output path or `GET /v1/runs/{run_id}/report`.

## Troubleshooting

- **Provider credential error at startup:** Confirm `MODEL_PROVIDER` matches the populated key.
  Blank values are treated as missing. For Bedrock, verify `aws sts get-caller-identity` succeeds in
  the same shell. Fallbacks are opt-in through `MODEL_FALLBACK_ORDER`.
- **OpenAI authentication error:** Check that `OPENAI_API_KEY` is a server-side API key available to
  the selected project and that `OPENAI_MODEL_ID` is enabled for it. The key is never sent in model
  prompts or logged.
- **Chromium executable missing:** Run `playwright install chromium` inside the active virtual
  environment. On Linux, Playwright may also require its documented system dependencies.
- **Structured-output failure:** Confirm the chosen model supports the configured Strands
  structured-output path. Inspect the safe role/provider/model log and select another accessible
  model ID; invalid model output receives one bounded repair attempt.
- **Tavily 401/403 or empty discovery:** Verify `TAVILY_API_KEY`, account quota, and outbound HTTPS.
  Search snippets are intentionally not accepted as evidence; Playwright must fetch a public page.
- **Public URL rejected:** The SSRF guard rejects credentials in URLs, nonstandard ports, redirects
  to private networks, localhost, and link-local addresses. This protection is not configurable.
- **Upload configuration failure:** Public-web mode intentionally uses `UPLOADS_ENABLED=false`. An
  approved upload workstream fails clearly until OpenSearch and the upload ingestion services are
  explicitly configured.
- **Run vanished after restart:** the default in-memory repository is process-local. Configure the
  existing DynamoDB/SQS production adapters when cross-process durability is required.

## Worker and dispatch

Starting a run, completing clarification, and approving a plan enqueue typed phase jobs. The
worker executes one bounded graph phase per delivery and checkpoints before the approval interrupt
and after every completed outer node or research workstream. A replacement worker resumes from the
canonical checkpoint; duplicate Standard-queue delivery is safe through optimistic revisions and
node-scoped idempotency keys.

Without `SQS_JOBS_QUEUE_URL`, dispatch uses an in-memory acknowledged queue. Use
`deep-research-dev` so API and worker share it. Separate production processes require SQS plus
DynamoDB and use `deep-research-worker`.

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
GET  /v1/runs/{run_id}/report
```

Every mutating request requires an `Idempotency-Key` header of 8–200 characters. Reusing a key for
the same command returns the original response; reusing it for different input returns `409`.
Plan approval requires the exact current plan version and canonical SHA-256 hash. Lifecycle updates
atomically persist the new run revision, one immutable ordered event, and the idempotency response.
Event polling accepts a cursor so clients can reconnect without losing progress. Worker events
contain only safe phase metadata, checksums, counts, and budget totals—not prompts or evidence text.
The report endpoint returns `text/markdown` only after completion, returns `409` while not ready,
and uses the same owner/tenant authorization as every other run endpoint.

The clarification endpoint accepts the checkpointed round and typed answers, for example
`{"round_number":1,"answers":[{"question_id":"audience","value":"Engineering leaders"}]}`.
Pending questions, the proposed brief, submitted answers, and the round are checkpointed before
each interrupt. Unknown/stale questions, duplicate answers, omitted required answers, and values of
the wrong answer type return `409`. A replay with the same idempotency key is safe.

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

The public-web bundle uses Tavily for discovery and an isolated Playwright Chromium context for
opened-page evidence. With `UPLOADS_ENABLED=false`, it does not construct OpenSearch, ClamAV, S3,
DynamoDB, or SQS services. If an approved plan nevertheless requests upload search, the run fails
with a typed configuration error instead of broadening its scope. When uploads are explicitly
enabled, a tenant/run-filtered OpenSearch adapter is constructed lazily. Tavily snippets remain
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
