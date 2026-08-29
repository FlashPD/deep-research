from deep_research.contracts.runs import Principal, RunEvent
from deep_research.services.runs import RunControlService


class RunEventPoller:
    """Reusable cursor tracker for local clients and the development CLI."""

    def __init__(self, *, after_cursor: int = 0, page_size: int = 100) -> None:
        if after_cursor < 0:
            raise ValueError("after_cursor cannot be negative")
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")
        self.cursor = after_cursor
        self.page_size = page_size

    async def poll(
        self,
        service: RunControlService,
        principal: Principal,
        run_id: str,
    ) -> list[RunEvent]:
        page = await service.get_events(
            principal,
            run_id,
            after_cursor=self.cursor,
            limit=self.page_size,
        )
        self.cursor = page.next_cursor
        return page.events
