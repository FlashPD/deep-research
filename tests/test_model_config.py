from pathlib import Path

from deep_research.models.config import ModelProvider, ModelSettings


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
