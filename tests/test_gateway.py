from types import SimpleNamespace
from typing import Any

import pytest

from deep_research.contracts.clarification import ClarificationDecision, ResearchBrief
from deep_research.models.config import ModelSettings
from deep_research.models.gateway import ModelGateway


class FakeAgent:
    calls = 0

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def invoke_async(self, *_args: Any, **_kwargs: Any) -> Any:
        type(self).calls += 1
        if type(self).calls == 1:
            raise TimeoutError("primary failed before producing a result")
        return SimpleNamespace(
            structured_output=ClarificationDecision(
                status="scope_ready",
                brief=ResearchBrief(topic="Fallback test"),
                interpretation_summary="The fallback completed the whole operation.",
            )
        )


@pytest.mark.asyncio
async def test_gateway_discards_failure_and_uses_next_provider(monkeypatch: Any) -> None:
    FakeAgent.calls = 0
    settings = ModelSettings.model_validate(
        {
            "targets": {
                "primary": {
                    "provider": "bedrock",
                    "model_id": "primary-model",
                    "max_attempts": 1,
                },
                "fallback": {
                    "provider": "anthropic",
                    "model_id": "fallback-model",
                    "max_attempts": 1,
                },
            },
            "roles": {"clarifier": ["primary", "fallback"]},
        }
    )
    monkeypatch.setattr(ModelGateway, "_create_model", staticmethod(lambda _target: object()))
    gateway = ModelGateway(settings, agent_factory=FakeAgent)

    result = await gateway.generate_structured(
        role="clarifier",
        prompt="Clarify this",
        output_type=ClarificationDecision,
        system_prompt="No tools",
    )

    assert result.brief.topic == "Fallback test"
    assert FakeAgent.calls == 2
