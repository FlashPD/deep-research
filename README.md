# Deep Research

Python 3.13 backend foundation for the architecture in
[`arch_plan/deep-research-system-plan.md`](arch_plan/deep-research-system-plan.md).

Implemented so far:

- typed clarification and normalized research-brief contracts;
- a three-round, no-tools Clarifier Agent with explicit final confirmation;
- a no-tools Planning Agent with validated workstream DAGs and fixed depth ceilings;
- deterministic plan versions and canonical SHA-256 approval hashes;
- a no-tools Questions Agent producing report-linked, prioritized follow-up topics;
- configuration-driven Bedrock, Anthropic, and OpenAI Strands model routing;
- whole-operation retries and provider fallback without mixing partial outputs;
- a minimal FastAPI clarification endpoint and health check;
- credential-free unit tests using fake structured model responses.

## Local setup

The checked-in package requires Python 3.13. A local `.venv` has been created with Python 3.13.15.

```bash
source .venv/bin/activate
pip install -e '.[dev]'
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
POST /v1/questions/generate
```

The request carries the topic, safe upload metadata, previous answers, current normalized brief,
and clarification round. The response is always a validated `ClarificationDecision`; the endpoint
does not expose public-web or retrieval tools.

The planning endpoint accepts an approved brief, depth preset, upload metadata, edits, and an
optional previous plan. Budget ceilings, version, and content hash are application-controlled. The
questions endpoint accepts a compact completed-report context and returns five to ten suggestions;
it never starts a new run.
