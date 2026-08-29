from collections.abc import Awaitable, Callable

CancellationCheck = Callable[[], Awaitable[object]]


async def check_cancellation(check: CancellationCheck | None) -> None:
    if check is not None:
        await check()
