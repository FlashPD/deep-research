from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from deep_research.models.config import ModelSettings


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Deep Research API"
    model_config_path: Path = Path("config/models.yaml")

    def load_models(self) -> ModelSettings:
        return ModelSettings.from_yaml(self.model_config_path)


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
