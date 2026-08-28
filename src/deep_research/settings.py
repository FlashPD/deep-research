from functools import lru_cache
from pathlib import Path

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

    def load_models(self) -> ModelSettings:
        return ModelSettings.from_yaml(self.model_config_path)


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
