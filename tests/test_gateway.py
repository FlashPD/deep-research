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


class BillingRejectedAgent:
    calls = 0

    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def invoke_async(self, *_args: Any, **_kwargs: Any) -> Any:
        type(self).calls += 1
        error_type = type("BadRequestError", (RuntimeError,), {})
        raise error_type(
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'Your credit balance is too low to access the Anthropic API.'}}"
        )


@pytest.mark.asyncio
async def test_gateway_does_not_retry_same_target_after_provider_rejection(
    monkeypatch: Any,
) -> None:
    from deep_research.models.gateway import is_provider_rejection_error

    BillingRejectedAgent.calls = 0
    settings = ModelSettings.model_validate(
        {
            "targets": {
                "primary": {
                    "provider": "anthropic",
                    "model_id": "primary-model",
                    "max_attempts": 3,
                },
                "fallback": {
                    "provider": "openai",
                    "model_id": "fallback-model",
                    "max_attempts": 3,
                },
            },
            "roles": {"researcher": ["primary", "fallback"]},
        }
    )
    monkeypatch.setattr(ModelGateway, "_create_model", staticmethod(lambda _target: object()))
    gateway = ModelGateway(settings, agent_factory=BillingRejectedAgent)

    with pytest.raises(ModelInvocationError) as exc_info:
        await gateway.generate_structured(
            role="researcher",
            prompt="Synthesize",
            output_type=ClarificationDecision,
            system_prompt="Return structured output",
        )

    # One attempt per target: the rejection is deterministic, but a fallback provider may
    # still succeed, so the gateway moves on instead of retrying the same target.
    assert BillingRejectedAgent.calls == 2
    assert [failure.target_name for failure in exc_info.value.failures] == ["primary", "fallback"]
    assert is_provider_rejection_error(exc_info.value)


def test_transient_failures_keep_the_error_retryable() -> None:
    from deep_research.models.config import ModelProvider
    from deep_research.models.gateway import (
        ModelAttemptFailure,
        is_provider_rejection_error,
        is_provider_rejection_failure,
    )

    def failure(error_type: str, message: str) -> ModelAttemptFailure:
        return ModelAttemptFailure(
            target_name="t",
            provider=ModelProvider.ANTHROPIC,
            model_id="m",
            attempt=1,
            max_attempts=2,
            error_type=error_type,
            message=message,
        )

    rate_limited = failure("RateLimitError", "Error code: 429 - rate_limit_error")
    overloaded = failure("InternalServerError", "Error code: 529 - overloaded_error")
    timed_out = failure("TimeoutError", "")
    rejected = failure("AuthenticationError", "Error code: 401 - invalid x-api-key")
    bedrock_denied = failure(
        "ClientError", "An error occurred (AccessDeniedException) when calling"
    )

    assert not is_provider_rejection_failure(rate_limited)
    assert not is_provider_rejection_failure(overloaded)
    assert not is_provider_rejection_failure(timed_out)
    assert is_provider_rejection_failure(rejected)
    assert is_provider_rejection_failure(bedrock_denied)
    assert not is_provider_rejection_error(ModelInvocationError("r", [rejected, rate_limited]))
    assert is_provider_rejection_error(ModelInvocationError("r", [rejected, bedrock_denied]))
    assert not is_provider_rejection_error(ModelInvocationError("r", []))
