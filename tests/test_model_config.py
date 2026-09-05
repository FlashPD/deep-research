from pathlib import Path

import pytest
from pydantic import SecretStr

from deep_research.models.config import ModelProvider, ModelSettings
from deep_research.settings import AppSettings


def test_loads_yaml_and_filters_unconfigured_optional_provider(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = tmp_path / "models.yaml"
    config.write_text(
        """
models:
  targets:
    primary:
      provider: bedrock
      model_id: ${MODEL_ID:-test-model}
    fallback:
      provider: openai
      model_id: test-openai
      required_env: OPENAI_API_KEY
  roles:
    clarifier: [primary, fallback]
""",
        encoding="utf-8",
    )

    settings = ModelSettings.from_yaml(config)

    assert settings.targets["primary"].provider is ModelProvider.BEDROCK
    assert [name for name, _ in settings.route_for("clarifier")] == ["primary"]


@pytest.mark.parametrize(
    ("provider", "key_field"),
    [("openai", "openai_api_key"), ("anthropic", "anthropic_api_key")],
)
def test_selected_direct_provider_is_first_and_receives_settings_key(
    tmp_path: Path, provider: str, key_field: str
) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        """
models:
  targets:
    bedrock: {provider: bedrock, model_id: bedrock-model}
    openai: {provider: openai, model_id: openai-model, required_env: OPENAI_API_KEY}
    anthropic: {provider: anthropic, model_id: anthropic-model, required_env: ANTHROPIC_API_KEY}
  roles:
    clarifier: [bedrock, openai, anthropic]
""",
        encoding="utf-8",
    )
    settings = AppSettings(
        model_config_path=config,
        model_provider=provider,
        **{key_field: SecretStr("from-dotenv-settings")},
    ).load_models()

    route = settings.route_for("clarifier")
    assert [target.provider.value for _, target in route] == [provider]
    assert route[0][1].api_key is not None
    assert route[0][1].api_key.get_secret_value() == "from-dotenv-settings"


def test_selected_direct_provider_fails_fast_without_key(tmp_path: Path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        """
models:
  targets:
    openai: {provider: openai, model_id: openai-model, required_env: OPENAI_API_KEY}
  roles:
    clarifier: [openai]
""",
        encoding="utf-8",
    )
    settings = AppSettings(model_config_path=config, model_provider="openai")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.validate_model_credentials()


def test_model_ids_from_dotenv_reach_the_yaml_routes(monkeypatch) -> None:
    # pydantic-settings never exports .env into os.environ, so the YAML's ${ANTHROPIC_MODEL_ID}
    # expansion must be fed from the loaded settings, not the process environment.
    monkeypatch.delenv("ANTHROPIC_MODEL_ID", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_ID", raising=False)
    settings = AppSettings(
        model_config_path=Path("config/models.yaml"),
        model_provider="anthropic",
        anthropic_api_key="sk-test",
        anthropic_model_id="model-from-dotenv",
        openai_model_id="openai-from-dotenv",
    )

    models = settings.load_models()

    assert models.targets["anthropic_direct"].model_id == "model-from-dotenv"
    assert models.targets["anthropic_report"].model_id == "model-from-dotenv"
    assert models.targets["openai_direct"].model_id == "openai-from-dotenv"
    assert models.targets["bedrock_default"].model_id.startswith("global.anthropic")


def test_every_target_can_emit_deep_sized_structured_output() -> None:
    settings = ModelSettings.from_yaml(Path("config/models.yaml"))

    # A deep plan or review is far larger than 4k tokens; every target's ceiling must fit it,
    # and the report targets get the longest timeout because they generate the most prose.
    for name, target in settings.targets.items():
        assert target.max_tokens >= 16_384, name
        assert target.timeout_seconds >= 240, name
    assert settings.targets["bedrock_report"].timeout_seconds == 300
    assert settings.targets["anthropic_report"].timeout_seconds == 300
    assert settings.targets["openai_report"].timeout_seconds == 300
    assert settings.roles["report"] == [
        "bedrock_report",
        "anthropic_report",
        "openai_report",
    ]
