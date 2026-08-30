from dataclasses import dataclass
from typing import Any

from strands.multiagent import GraphBuilder

from deep_research.contracts.planning import ResearchPlan


@dataclass
class _TypedNode:
    """Topology-only Strands executor; DurableWorker invokes the typed boundaries."""

    id: str

    async def invoke_async(self, prompt: Any = None, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("durable graph nodes are invoked by DurableWorker")

    async def stream_async(self, prompt: Any = None, **kwargs: Any):  # pragma: no cover
        yield await self.invoke_async(prompt, **kwargs)

    def __call__(self, prompt: Any = None, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("use DurableWorker for persisted graph execution")


class BoundedResearchGraph:
    """The bounded Strands topology paired with the durable checkpoint state machine."""

    MAX_REPAIR_ROUNDS = 2
    MAX_NODE_EXECUTIONS = 12

    def __init__(self, *, execution_timeout_seconds: float = 2_700) -> None:
        builder = GraphBuilder()
        for node_id in (
            "clarifier",
            "planner",
            "plan_approval_interrupt",
            "research",
            "evidence_reviewer",
            "report_generator",
            "questions_agent",
            "finalization",
        ):
            builder.add_node(_TypedNode(node_id), node_id)
        builder.add_edge("clarifier", "planner")
        builder.add_edge("planner", "plan_approval_interrupt")
        builder.add_edge("plan_approval_interrupt", "research")
        builder.add_edge("research", "evidence_reviewer")
        builder.add_edge(
            "evidence_reviewer", "research", condition=_repair_required
        )
        builder.add_edge(
            "evidence_reviewer", "report_generator", condition=_review_accepted
        )
        builder.add_edge("report_generator", "questions_agent")
        builder.add_edge("questions_agent", "finalization")
        builder.set_entry_point("clarifier")
        builder.set_max_node_executions(self.MAX_NODE_EXECUTIONS)
        builder.set_execution_timeout(execution_timeout_seconds)
        builder.set_node_timeout(300)
        self.strands_graph = builder.build()

    @staticmethod
    def build_research_graph(plan: ResearchPlan):
        """Create the approved workstream DAG; each workstream has an isolated executor."""
        builder = GraphBuilder()
        for workstream in plan.workstreams:
            builder.add_node(_TypedNode(workstream.id), workstream.id)
        for workstream in plan.workstreams:
            for dependency in workstream.dependencies:
                builder.add_edge(dependency, workstream.id)
        builder.set_max_node_executions(len(plan.workstreams))
        builder.set_execution_timeout(plan.budget.target_duration_seconds)
        builder.set_node_timeout(300)
        return builder.build()


def _repair_required(state: Any, invocation_state: dict[str, Any]) -> bool:
    return bool(invocation_state.get("repair_required"))


def _review_accepted(state: Any, invocation_state: dict[str, Any]) -> bool:
    return not _repair_required(state, invocation_state) and not bool(
        invocation_state.get("review_rejected")
    )
