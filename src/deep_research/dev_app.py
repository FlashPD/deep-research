import logging
from dataclasses import dataclass

import uvicorn

from deep_research.api.app import create_app
from deep_research.jobs.memory import InMemoryJobDispatcher
from deep_research.models.gateway import ModelGateway, StructuredModelGateway
from deep_research.persistence.memory import InMemoryRunRepository
from deep_research.services.runs import RunControlService
from deep_research.settings import AppSettings, get_settings
from deep_research.worker import DurableWorker
from deep_research.worker_app import create_worker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalDevelopmentRuntime:
    settings: AppSettings
    repository: InMemoryRunRepository
    dispatcher: InMemoryJobDispatcher
    control: RunControlService
    gateway: StructuredModelGateway
    worker: DurableWorker


def build_local_runtime(settings: AppSettings | None = None) -> LocalDevelopmentRuntime:
    """Compose shared local services without starting either UI surface."""
    configured = settings or get_settings()
    models = configured.validate_model_credentials()
    if (
        configured.tavily_api_key is None
        or not configured.tavily_api_key.get_secret_value().strip()
    ):
        raise ValueError("TAVILY_API_KEY is required for the development runtime")

    repository = InMemoryRunRepository()
    dispatcher = InMemoryJobDispatcher()
    control = RunControlService(repository, retention_days=configured.run_retention_days)
    gateway = ModelGateway(models)
    worker = create_worker(
        settings=configured,
        gateway=gateway,
        run_repository=repository,
        job_dispatcher=dispatcher,
        run_control=control,
    )
    return LocalDevelopmentRuntime(
        settings=configured,
        repository=repository,
        dispatcher=dispatcher,
        control=control,
        gateway=gateway,
        worker=worker,
    )


def create_dev_app(settings: AppSettings | None = None):
    """Compose the complete local vertical slice around shared in-memory adapters."""
    runtime = build_local_runtime(settings)
    logger.info("local runtime provider=%s", runtime.settings.model_provider.value)
    return create_app(
        run_repository=runtime.repository,
        job_dispatcher=runtime.dispatcher,
        run_control=runtime.control,
        background_worker=runtime.worker,
        expose_agent_debug_routes=False,
    )


def main() -> None:
    uvicorn.run(create_dev_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
