# Deep Research

Initial Python 3.13 backend slice for the architecture in
[`arch_plan/deep-research-system-plan.md`](arch_plan/deep-research-system-plan.md).

Implemented so far:

- typed clarification and normalized research-brief contracts;
- a three-round, no-tools Clarifier Agent with explicit final confirmation;
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

## Clarifier endpoint

```text
POST /v1/clarifier/evaluate
```

The request carries the topic, safe upload metadata, previous answers, current normalized brief,
and clarification round. The response is always a validated `ClarificationDecision`; the endpoint
does not expose public-web or retrieval tools.
