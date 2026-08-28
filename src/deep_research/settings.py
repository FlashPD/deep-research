from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from deep_research.models.config import ModelSettings


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Deep Research API"
    model_config_path: Path = Path("config/models.yaml")
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
    auth_mode: Literal["development", "cognito"] = "development"
    cognito_issuer: str | None = None
    cognito_client_id: str | None = None
    cognito_required_scopes: str = "research:run"
    cognito_tenant_claim: str = "custom:tenant_id"
    cognito_jwks_cache_seconds: int = 3_600
    run_retention_days: int = 30
    dynamodb_runs_table: str | None = None
    aws_region: str = "us-east-1"

    def load_models(self) -> ModelSettings:
        return ModelSettings.from_yaml(self.model_config_path)

    @property
    def required_scopes(self) -> frozenset[str]:
        return frozenset(self.cognito_required_scopes.split())


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
