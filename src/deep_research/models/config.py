import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelProvider(StrEnum):
    BEDROCK = "bedrock"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ModelProvider
    model_id: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1)
    timeout_seconds: float = Field(default=90, gt=0)
    max_attempts: int = Field(default=2, ge=1, le=5)
    region: str | None = None
    base_url: str | None = None
    required_env: str | None = None

    @property
    def is_available(self) -> bool:
        return self.required_env is None or bool(os.getenv(self.required_env))


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: dict[str, ModelTarget]
    roles: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_routes(self) -> "ModelSettings":
        missing = {
            target_name
            for route in self.roles.values()
            for target_name in route
            if target_name not in self.targets
        }
        if missing:
            raise ValueError(f"role routes reference unknown model targets: {sorted(missing)}")
        if any(not route for route in self.roles.values()):
            raise ValueError("every role must contain at least one model target")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelSettings":
        raw = Path(path).read_text(encoding="utf-8")
        expanded = _expand_environment(raw)
        document: dict[str, Any] = yaml.safe_load(expanded)
        return cls.model_validate(document["models"])

    def route_for(self, role: str) -> list[tuple[str, ModelTarget]]:
        try:
            names = self.roles[role]
        except KeyError as exc:
            raise ValueError(f"no model route configured for role {role!r}") from exc
        available = [
            (name, self.targets[name]) for name in names if self.targets[name].is_available
        ]
        if not available:
            raise ValueError(f"no available model targets configured for role {role!r}")
        return available


_ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?}")


def _expand_environment(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.getenv(name, default)
        if resolved is None:
            raise ValueError(f"required environment variable {name} is not set")
        return resolved

    return _ENV_PATTERN.sub(replace, value)
