"""Groq chat client: shared HTTP connection, key rotation, streaming + usage."""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from ..config import Settings
from ..utils.errors import RagError
from ..utils.logging import log_event
from .key_rotator import KeyRotator

_FRIENDLY = {
    401: "The Groq API key was rejected. Check GROQ_API_KEY in .env.",
    403: "Groq refused this request for the configured key.",
    404: "The selected model is not available on Groq for this key.",
    429: "Groq is rate limiting requests. Please retry in a moment.",
}


class GroqClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rotator = KeyRotator(settings.groq_keys, settings.requests_per_key, settings.key_cooldown_seconds)
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.settings.groq_base_url, timeout=self.settings.groq_timeout
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def validate_model(self, model: str | None) -> str:
        if not model:
            return self.settings.model_20b
        if model not in self.settings.allowed_models:
            raise RagError(f"Model '{model}' is not enabled. Allowed: {', '.join(self.settings.allowed_models)}.", 400)
        return model

    async def _payload(self, messages, model, stream, **kwargs) -> tuple[dict, dict]:
        key = await self.rotator.acquire()
        body = {"model": model, "messages": messages, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
        body.update({k: v for k, v in kwargs.items() if v is not None})
        return body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _fail(self, status: int, detail: str) -> RagError:
        log_event("groq_error", status=status)
        return RagError(_FRIENDLY.get(status, "The language model service failed. Please retry."), 502, detail=detail)

    async def complete(self, messages: list[dict], model: str, **kwargs) -> tuple[str, dict]:
        if self._client is None:
            await self.startup()
        body, headers = await self._payload(messages, model, False, **kwargs)
        try:
            res = await self._client.post("/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise RagError("Could not reach the language model service.", 504, detail=str(exc))
        if res.status_code >= 400:
            raise self._fail(res.status_code, res.text[:500])
        try:
            data = res.json()
            return data["choices"][0]["message"]["content"] or "", data.get("usage") or {}
        except (ValueError, KeyError, IndexError) as exc:
            raise RagError("The language model returned an unreadable response.", 502, detail=str(exc))

    async def stream(self, messages: list[dict], model: str, **kwargs) -> AsyncIterator[dict]:
        """Yields {'delta': str} chunks then {'usage': {...}}."""
        if self._client is None:
            await self.startup()
        body, headers = await self._payload(messages, model, True, **kwargs)
        usage: dict = {}
        try:
            async with self._client.stream("POST", "/chat/completions", json=body, headers=headers) as res:
                if res.status_code >= 400:
                    raise self._fail(res.status_code, (await res.aread()).decode()[:500])
                async for line in res.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except ValueError:
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            yield {"delta": delta}
        except httpx.HTTPError as exc:
            raise RagError("The answer stream was interrupted.", 504, detail=str(exc))
        yield {"usage": usage}
