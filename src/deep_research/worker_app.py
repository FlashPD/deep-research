import asyncio

from deep_research.agents import (
    ClarifierAgent,
    EvidenceReviewer,
    PlanningAgent,
    QuestionsAgent,
    ReportGenerationAgent,
    ResearchAgent,
)
from deep_research.jobs import InMemoryJobDispatcher, JobDispatcher, SQSJobDispatcher
from deep_research.models.gateway import ModelGateway, StructuredModelGateway
from deep_research.persistence import (
    DynamoDBRunRepository,
    InMemoryRunRepository,
    RunRepository,
)
from deep_research.services.runs import RunControlService
from deep_research.settings import AppSettings, get_settings
from deep_research.tools.live import build_live_research_services
from deep_research.worker import DurableWorker, WorkerAgents


def create_worker(
    *,
    settings: AppSettings | None = None,
    gateway: StructuredModelGateway | None = None,
    run_repository: RunRepository | None = None,
    job_dispatcher: JobDispatcher | None = None,
    run_control: RunControlService | None = None,
) -> DurableWorker:
    """Compose a worker; injected local adapters can be shared with the API in tests/dev."""
    configured = settings or get_settings()
    model_gateway = gateway or ModelGateway(configured.validate_model_credentials())
    repository = run_repository or _build_repository(configured)
    dispatcher = job_dispatcher or _build_dispatcher(configured)

    def researcher_factory(principal, run_id: str) -> ResearchAgent:
        services = build_live_research_services(
            configured,
            tenant_id=principal.tenant_id,
            run_id=run_id,
        )
        return ResearchAgent(model_gateway, services.research_adapter)

    return DurableWorker(
        run_control or RunControlService(repository, retention_days=configured.run_retention_days),
        dispatcher,
        WorkerAgents(
            clarifier=ClarifierAgent(model_gateway),
            planner=PlanningAgent(model_gateway),
            researcher_factory=researcher_factory,
            reviewer=EvidenceReviewer(model_gateway),
            report=ReportGenerationAgent(model_gateway),
            questions=QuestionsAgent(model_gateway),
        ),
    )


def _build_repository(settings: AppSettings) -> RunRepository:
    if not settings.dynamodb_runs_table:
        return InMemoryRunRepository()
    import boto3

    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.dynamodb_runs_table
    )
    return DynamoDBRunRepository(table)


def _build_dispatcher(settings: AppSettings) -> JobDispatcher:
    if not settings.sqs_jobs_queue_url:
        return InMemoryJobDispatcher()
    import boto3

    return SQSJobDispatcher(
        boto3.client("sqs", region_name=settings.aws_region),
        settings.sqs_jobs_queue_url,
    )


def main() -> None:
    asyncio.run(create_worker().run_forever())


if __name__ == "__main__":  # pragma: no cover
    main()
