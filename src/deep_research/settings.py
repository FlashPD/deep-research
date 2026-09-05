from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from deep_research.models.config import ModelProvider, ModelSettings


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Deep Research API"
    model_config_path: Path = Path("config/models.yaml")
    model_provider: ModelProvider = ModelProvider.BEDROCK
    model_fallback_order: str = ""
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    # Model IDs are consumed by config/models.yaml through ${VAR:-default} expansion. They are
    # declared here so values in .env reach the YAML (see load_models) instead of being ignored.
    openai_model_id: str | None = None
    anthropic_model_id: str | None = None
    default_model_id: str | None = None
    tavily_api_key: SecretStr | None = None
    tavily_base_url: str = "https://api.tavily.com"
    playwright_headless: bool = True
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "deep-research-upload-chunks"
    opensearch_username: str | None = None
    opensearch_password: SecretStr | None = None
    opensearch_aws_region: str | None = None
    opensearch_aws_service: str = "aoss"
    upload_artifact_root: Path = Path(".data/uploads")
    clamav_host: str = "localhost"
    clamav_port: int = 3310
    uploads_enabled: bool = False
    auth_mode: Literal["development", "cognito"] = "development"
    cognito_issuer: str | None = None
    cognito_client_id: str | None = None
    cognito_required_scopes: str = "research:run"
    cognito_tenant_claim: str = "custom:tenant_id"
    cognito_jwks_cache_seconds: int = 3_600
    run_retention_days: int = 30
    dynamodb_runs_table: str | None = None
    sqs_jobs_queue_url: str | None = None
    aws_region: str = "us-east-1"

    def load_models(self) -> ModelSettings:
        yaml_overrides = {
            name: value
            for name, value in {
                "OPENAI_MODEL_ID": self.openai_model_id,
                "ANTHROPIC_MODEL_ID": self.anthropic_model_id,
                "DEFAULT_MODEL_ID": self.default_model_id,
                "AWS_REGION": self.aws_region,
            }.items()
            if value
        }
        configured = ModelSettings.from_yaml(self.model_config_path, overrides=yaml_overrides)
        fallbacks = [
            ModelProvider(item.strip())
            for item in self.model_fallback_order.split(",")
            if item.strip()
        ]
        return configured.select_provider(
            self.model_provider,
            fallback_order=fallbacks,
            api_keys={
                ModelProvider.OPENAI: self.openai_api_key,
                ModelProvider.ANTHROPIC: self.anthropic_api_key,
            },
        )

    def validate_model_credentials(self) -> ModelSettings:
        models = self.load_models()
        selected = self.model_provider
        if selected is ModelProvider.OPENAI and not _secret_is_set(self.openai_api_key):
            raise ValueError("MODEL_PROVIDER=openai requires OPENAI_API_KEY")
        if selected is ModelProvider.ANTHROPIC and not _secret_is_set(self.anthropic_api_key):
            raise ValueError("MODEL_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        if selected is ModelProvider.BEDROCK:
            import boto3

            if boto3.Session(region_name=self.aws_region).get_credentials() is None:
                raise ValueError(
                    "MODEL_PROVIDER=bedrock requires credentials in the standard AWS "
                    "credential chain"
                )
        for role in models.roles:
            models.route_for(role)
        return models

    @property
    def required_scopes(self) -> frozenset[str]:
        return frozenset(self.cognito_required_scopes.split())


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()


def _secret_is_set(value: SecretStr | None) -> bool:
    return value is not None and bool(value.get_secret_value().strip())
