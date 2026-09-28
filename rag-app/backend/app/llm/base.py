from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, messages: list[dict], model: str, **kwargs) -> tuple[str, dict]: ...

    def stream(self, messages: list[dict], model: str, **kwargs) -> AsyncIterator[dict]: ...
