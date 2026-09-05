from collections import deque
from typing import Any

from pydantic import BaseModel


class FakeGateway:
    def __init__(self, *responses: BaseModel | BaseException) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(self, **kwargs: Any) -> BaseModel:
        self.calls.append(kwargs)
        response = self.responses.popleft()
        if isinstance(response, BaseException):
            raise response
        return response
