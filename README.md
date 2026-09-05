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
- a phase-scoped page cache so parallel workstreams share one capture per URL, plus tolerant
  evidence merging when a dynamic page hashes differently between fetches;
- fail-fast worker classification for deterministic failures (validation errors, model output
  limits, provider billing/credential rejections) so a doomed phase is never redelivered;
- graceful report degradation: over-length quick prose is trimmed, unsafe diagrams become prose,
  and a still-invalid draft falls back to a deterministic, fully cited evidence-mapped report;
- credential-free mocked tests plus explicitly selected, bounded live tests.

Two completed runs are checked in as examples; see [Sample reports](#sample-reports).

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
are passed directly to the matching Strands client and are never logged. The model-ID variables
(`OPENAI_MODEL_ID`, `ANTHROPIC_MODEL_ID`, `DEFAULT_MODEL_ID`, `AWS_REGION`) are read from `.env`
too and substituted into the YAML's `${VAR:-default}` placeholders.

Every target allows 16k output tokens with a 4-to-5-minute timeout. Deep-depth plans, syntheses,
and reviews are large structured documents; a smaller ceiling makes the model stop mid-output and
the run fails with `model_output_limit`.

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

- `--depth quick|standard|deep` (default `standard`)
- `--provider openai|anthropic|bedrock`, overriding `MODEL_PROVIDER` for that invocation
- `--output PATH`, where a `.md` path is used directly and a directory receives `<run_id>.md`
  (default directory `research-reports/`)
- `--auto-approve-plan`, which is the only way to bypass the interactive approval prompt
- `--verbose`, which logs model routing, provider fallbacks, and adapter failures to stderr

Depth presets are fixed by the application and appear in the plan the CLI prints for approval:

| Preset | Target duration | Search queries | Accepted sources | Parallel workstreams | Review repair rounds |
|---|---|---|---|---|---|
| `quick` | 5 min | 5 (+1 adaptive) | 10 | 3 | 1 |
| `standard` | 15 min | 20 | 30 | 6 | 2 |
| `deep` | 20 min | 50 | 75 | 10 | 2 |

Quick runs typically finish in 6 to 9 minutes of wall time, deep runs in about 30 minutes. A
deep run sends on the order of a million input tokens to the model provider (each workstream's
synthesis carries up to 400k characters of fetched text), so keep credit headroom accordingly.

The default flow displays typed clarification questions, submits answers for the current round,
prints the generated plan, asks for approval of its exact version and SHA-256 hash, polls safe
progress events, writes the completed Markdown report, and displays limitations and follow-up
questions. Press Ctrl-C while work is running to persist a cooperative cancellation request.

When stdin is not a terminal (piped input, CI, an editor task runner), the CLI refuses to start
unless `--auto-approve-plan` is given, and cancels the run with exit code 2 if a clarification
question still needs an answer. Parallel workstreams share one page capture per canonical URL
for the duration of a research phase, so a page that appears in several candidate queries is
fetched once and counted once per workstream.

Reviewing the plan is worth the pause: rejecting it cancels the run, so if the planner has
misidentified a product (for example, treating a model name as its predecessor), restart with the
exact model numbers in the topic rather than approving and hoping the reviewer catches it.

### Sample reports

Two completed runs are committed under [`reports/`](reports/) as examples of the output format:

- [`reports/Quick research sample report.md`](reports/Quick%20research%20sample%20report.md):
  `--depth quick`, topic "Give me a summary of Samsung's S90D TV". About 6.5 minutes, 2
  workstreams, 5 cited sources, one review repair round.
- [`reports/Deep research sample report.md`](reports/Deep%20research%20sample%20report.md):
  `--depth deep`, a three-way comparison of the Samsung S90D, LG C4, and Sony Bravia 8 with the
  exact model numbers pinned in the topic. About 32 minutes, 6 parallel workstreams, 41 searches,
  46 fetched sources, 225 reviewed claims, 36 cited sources.

Both reports were produced by the deterministic fallback path: the model's prose draft failed a
citation check twice (an uncited executive-summary paragraph in the quick run, multi-source
brackets like `[S1, S2]` in the deep run, the latter now normalized before validation), so the
report agent assembled the sections directly from reviewed, evidence-backed claims. The
`Limitations` section of each report states this. Every `[S…]` citation resolves to an entry in
the report's `Sources` section, and reviewer limitations, unresolved contradictions, and
follow-up topics are preserved. New runs write to the `--output` directory; `reports/` ignores
everything except the two samples.

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
   proceeding requires explicit final confirmation. Quick depth asks at most two questions and
   proceeds on the best available interpretation after the second round.
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
  structured-output path. Inspect the safe role/provider/model log (`--verbose`) and select
  another accessible model ID; invalid model output receives one bounded repair attempt, after
  which the researcher drops the invalid claims and the report agent trims or falls back rather
  than failing the run.
- **Run failed with `model_output_limit`:** the model stopped at its `max_tokens` ceiling. Raise
  `max_tokens` (and `timeout_seconds`) for that target in `config/models.yaml`; the run is not
  retried because the same prompt would hit the same ceiling.
- **Run failed with `model_provider_rejected`:** the provider refused the request for a reason
  that will not change on retry, most often an exhausted credit balance or an invalid key. The
  failure message carries the provider's text. Top up or fix the key and start a new run; nothing
  was redelivered, so no search or fetch budget was re-spent.
- **Run failed with `worker_validation_error`:** an agent's output or a checkpoint failed a
  deterministic check on the first delivery. The message names the check; it is a code or prompt
  issue, not a transient one.
- **Run failed with `insufficient_evidence`:** no workstream produced an evidence-backed claim,
  so there is nothing to report. Check the printed limitations for fetch failures (timeouts, 403s,
  paywalls) and broaden the topic or fix connectivity.
- **Report says it was "assembled deterministically":** the model's prose draft failed a hard
  citation or outline check twice, so the report was built directly from reviewed claims. It is
  complete and fully cited but reads as a claim list per section. See [Sample reports](#sample-reports).
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
