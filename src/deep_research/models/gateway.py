import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from deep_research.models.config import ModelProvider, ModelSettings, ModelTarget

OutputT = TypeVar("OutputT", bound=BaseModel)
logger = logging.getLogger(__name__)

_MAX_ERROR_DETAIL_CHARS = 500
_OUTPUT_LIMIT_ERROR_TYPES = {
    "LengthFinishReasonError",
    "MaxTokensReachedException",
}


class StructuredModelGateway(Protocol):
    async def generate_structured(
        self,
        *,
        role: str,
        prompt: str,
        output_type: type[OutputT],
        system_prompt: str,
    ) -> OutputT: ...


@dataclass(frozen=True)
class ModelAttemptFailure:
    target_name: str
    provider: ModelProvider
    model_id: str
    attempt: int
    max_attempts: int
    error_type: str
    message: str

    def summary(self) -> str:
        target = f"{self.target_name}[{self.provider.value}/{self.model_id}]"
        detail = f": {self.message}" if self.message else ""
        return (
            f"{target} attempt {self.attempt}/{self.max_attempts}: "
            f"{self.error_type}{detail}"
        )


class ModelInvocationError(RuntimeError):
    def __init__(self, role: str, failures: list[ModelAttemptFailure]) -> None:
        detail = "; ".join(failure.summary() for failure in failures)
        super().__init__(f"all model targets failed for role {role!r}: {detail}")
        self.role = role
        self.failures = tuple(failures)


class ModelGateway:
    """Role-aware Strands model factory with whole-operation fallback.

    A failed attempt is discarded in full. Outputs are never combined across providers.
    """

    def __init__(
        self,
        settings: ModelSettings,
        agent_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._settings = settings
        self._agent_factory = agent_factory

    async def generate_structured(
        self,
        *,
        role: str,
        prompt: str,
        output_type: type[OutputT],
        system_prompt: str,
    ) -> OutputT:
        failures: list[ModelAttemptFailure] = []
        for target_name, target in self._settings.route_for(role):
            logger.info(
                "model route selected role=%s provider=%s model=%s",
                role,
                target.provider.value,
                target.model_id,
            )
            for attempt_index in range(target.max_attempts):
                try:
                    return await asyncio.wait_for(
                        self._invoke(target, prompt, output_type, system_prompt),
                        timeout=target.timeout_seconds,
                    )
                except (Exception, asyncio.CancelledError) as exc:
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    failure = ModelAttemptFailure(
                        target_name=target_name,
                        provider=target.provider,
                        model_id=target.model_id,
                        attempt=attempt_index + 1,
                        max_attempts=target.max_attempts,
                        error_type=type(exc).__name__,
                        message=_safe_error_detail(exc),
                    )
                    failures.append(failure)
                    logger.warning(
                        "model invocation failed role=%s target=%s provider=%s model=%s "
                        "attempt=%s/%s error=%s detail=%s",
                        role,
                        target_name,
                        target.provider.value,
                        target.model_id,
                        failure.attempt,
                        failure.max_attempts,
                        failure.error_type,
                        failure.message or "<no detail>",
                    )
                    logger.debug("model invocation traceback", exc_info=True)
                    if is_output_limit_failure(failure):
                        break
        raise ModelInvocationError(role, failures)

    async def _invoke(
        self,
        target: ModelTarget,
        prompt: str,
        output_type: type[OutputT],
        system_prompt: str,
    ) -> OutputT:
        agent_factory = self._agent_factory
        if agent_factory is None:
            from strands import Agent

            agent_factory = Agent

        agent = agent_factory(
            model=self._create_model(target),
            system_prompt=system_prompt,
            tools=[],
        )
        result = await agent.invoke_async(prompt, structured_output_model=output_type)
        structured = result.structured_output
        if isinstance(structured, output_type):
            return structured
        return output_type.model_validate(structured)

    @staticmethod
    def _create_model(target: ModelTarget) -> Any:
        if target.provider is ModelProvider.BEDROCK:
            from botocore.config import Config as BotocoreConfig
            from strands.models import BedrockModel

            return BedrockModel(
                model_id=target.model_id,
                region_name=target.region,
                temperature=target.temperature,
                max_tokens=target.max_tokens,
                boto_client_config=BotocoreConfig(
                    connect_timeout=min(10, target.timeout_seconds),
                    read_timeout=target.timeout_seconds,
                    retries={"max_attempts": target.max_attempts, "mode": "standard"},
                ),
            )
        if target.provider is ModelProvider.OPENAI:
            from strands.models.openai import OpenAIModel

            client_args = {"timeout": target.timeout_seconds, "max_retries": 0}
            if target.api_key is not None:
                client_args["api_key"] = target.api_key.get_secret_value()
            if target.base_url:
                client_args["base_url"] = target.base_url
            return OpenAIModel(
                client_args=client_args,
                model_id=target.model_id,
                params={
                    "max_tokens": target.max_tokens,
                    "temperature": target.temperature,
                },
            )
        if target.provider is ModelProvider.ANTHROPIC:
            from strands.models.anthropic import AnthropicModel

            client_args = {"timeout": target.timeout_seconds, "max_retries": 0}
            if target.api_key is not None:
                client_args["api_key"] = target.api_key.get_secret_value()
            return AnthropicModel(
                client_args=client_args,
                model_id=target.model_id,
                max_tokens=target.max_tokens,
                params={"temperature": target.temperature},
            )
        raise AssertionError(f"unsupported model provider: {target.provider}")


def _safe_error_detail(exc: BaseException) -> str:
    """Keep provider diagnostics readable and bounded without logging request payloads."""
    detail = " ".join(str(exc).split())
    if len(detail) <= _MAX_ERROR_DETAIL_CHARS:
        return detail
    return f"{detail[: _MAX_ERROR_DETAIL_CHARS - 3]}..."


def is_output_limit_failure(failure: ModelAttemptFailure) -> bool:
    detail = failure.message.casefold()
    return failure.error_type in _OUTPUT_LIMIT_ERROR_TYPES or (
        "maximum token" in detail or "max token" in detail
    )


def is_output_limit_error(error: ModelInvocationError) -> bool:
    return bool(error.failures) and any(
        is_output_limit_failure(failure) for failure in error.failures
    )
