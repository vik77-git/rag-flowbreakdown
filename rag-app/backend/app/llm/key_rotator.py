"""Round-robin Groq key rotation: N requests per key, then a cooldown."""
from __future__ import annotations

import asyncio
import time

from ..utils.errors import RagError


class KeyRotator:
    def __init__(self, keys: list[str], requests_per_key: int = 5, cooldown_seconds: float = 30.0):
        self.keys = [k for k in keys if k]
        self.requests_per_key = max(1, requests_per_key)
        self.cooldown = max(0.0, cooldown_seconds)
        self._used = [0] * len(self.keys)
        self._ready_at = [0.0] * len(self.keys)
        self._index = 0
        self._lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.keys)

    async def acquire(self) -> str:
        if not self.keys:
            raise RagError("No Groq API key is configured on the server. Add GROQ_API_KEY to .env.", 503)
        while True:
            async with self._lock:
                now = time.monotonic()
                for offset in range(len(self.keys)):
                    i = (self._index + offset) % len(self.keys)
                    if self._ready_at[i] > now:
                        continue
                    self._used[i] += 1
                    if self._used[i] >= self.requests_per_key:
                        self._used[i] = 0
                        self._ready_at[i] = now + self.cooldown
                        self._index = (i + 1) % len(self.keys)
                    return self.keys[i]
                wait = min(self._ready_at) - now
            await asyncio.sleep(min(max(wait, 0.05), self.cooldown or 1.0))

    def status(self) -> list[dict]:
        now = time.monotonic()
        return [
            {
                "key": f"key_{i + 1}",
                "requests_in_window": self._used[i],
                "cooling_down": self._ready_at[i] > now,
                "ready_in_seconds": max(0.0, round(self._ready_at[i] - now, 1)),
            }
            for i in range(len(self.keys))
        ]
