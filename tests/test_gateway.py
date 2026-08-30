from types import SimpleNamespace
from typing import Any

import pytest

from deep_research.contracts.clarification import ClarificationDecision, ResearchBrief
from deep_research.models.config import ModelSettings
from deep_research.models.gateway import ModelGateway, ModelInvocationError


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


class AlwaysFailingAgent:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def invoke_async(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("structured output omitted source_scores")


@pytest.mark.asyncio
async def test_gateway_failure_exposes_target_attempt_and_error_detail(
    monkeypatch: Any,
) -> None:
    settings = ModelSettings.model_validate(
        {
            "targets": {
                "reviewer_primary": {
                    "provider": "anthropic",
                    "model_id": "reviewer-model",
                    "max_attempts": 2,
                }
            },
            "roles": {"reviewer": ["reviewer_primary"]},
        }
    )
    monkeypatch.setattr(ModelGateway, "_create_model", staticmethod(lambda _target: object()))
    gateway = ModelGateway(settings, agent_factory=AlwaysFailingAgent)

    with pytest.raises(ModelInvocationError) as exc_info:
        await gateway.generate_structured(
            role="reviewer",
            prompt="Review evidence",
            output_type=ClarificationDecision,
            system_prompt="Return structured output",
        )

    message = str(exc_info.value)
    assert "reviewer_primary[anthropic/reviewer-model] attempt 1/2" in message
    assert "reviewer_primary[anthropic/reviewer-model] attempt 2/2" in message
    assert "ValueError: structured output omitted source_scores" in message
    assert [failure.attempt for failure in exc_info.value.failures] == [1, 2]


class OutputLimitedAgent:
    calls = 0

    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def invoke_async(self, *_args: Any, **_kwargs: Any) -> Any:
        type(self).calls += 1
        error_type = type("MaxTokensReachedException", (RuntimeError,), {})
        raise error_type("Model stopped at the maximum token limit")


@pytest.mark.asyncio
async def test_gateway_does_not_retry_same_target_after_output_limit(
    monkeypatch: Any,
) -> None:
    OutputLimitedAgent.calls = 0
    settings = ModelSettings.model_validate(
        {
            "targets": {
                "report_target": {
                    "provider": "anthropic",
                    "model_id": "report-model",
                    "max_attempts": 3,
                }
            },
            "roles": {"report": ["report_target"]},
        }
    )
    monkeypatch.setattr(ModelGateway, "_create_model", staticmethod(lambda _target: object()))
    gateway = ModelGateway(settings, agent_factory=OutputLimitedAgent)

    with pytest.raises(ModelInvocationError) as exc_info:
        await gateway.generate_structured(
            role="report",
            prompt="Generate report",
            output_type=ClarificationDecision,
            system_prompt="Return structured output",
        )

    assert OutputLimitedAgent.calls == 1
    assert len(exc_info.value.failures) == 1
