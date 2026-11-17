"""Minimal async client for Ollama's native REST API.

Endpoints used: GET /api/version, GET /api/tags, POST /api/show,
POST /api/chat (NDJSON stream) and POST /api/embed.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, host: str, transport: httpx.AsyncBaseTransport | None = None):
        self.host = host
        self.http = httpx.AsyncClient(
            base_url=host, transport=transport,
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=5.0))
        self._caps: dict[str, list[str]] = {}

    async def close(self) -> None:
        await self.http.aclose()

    async def _json(self, method: str, url: str, **kw) -> dict:
        try:
            r = await self.http.request(method, url, **kw)
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host} ({type(exc).__name__}). "
                "Is `ollama serve` running?") from exc
        if r.status_code >= 400:
            raise OllamaError(_error_text(r.status_code, r.text))
        return r.json()

    async def version(self) -> str:
        return (await self._json("GET", "/api/version")).get("version", "")

    async def models(self) -> list[str]:
        data = await self._json("GET", "/api/tags")
        return sorted(m.get("name") or m.get("model") for m in data.get("models", []))

    async def capabilities(self, model: str) -> list[str]:
        """Capabilities reported by /api/show, e.g. completion, tools, thinking, vision."""
        if model not in self._caps:
            try:
                data = await self._json("POST", "/api/show", json={"model": model})
                self._caps[model] = list(data.get("capabilities") or [])
            except OllamaError:
                return []
        return self._caps[model]

    async def embed(self, model: str, texts: list[str], keep_alive: str = "30m") -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            data = await self._json("POST", "/api/embed", json={
                "model": model, "input": texts[i:i + 32], "keep_alive": keep_alive})
            out.extend(data["embeddings"])
        return out

    async def chat(self, payload: dict[str, Any]) -> dict:
        return await self._json("POST", "/api/chat", json={**payload, "stream": False})

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict]:
        """Yield the NDJSON objects of a streaming /api/chat call."""
        try:
            async with self.http.stream("POST", "/api/chat", json={**payload, "stream": True}) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode("utf-8", "replace")
                    raise OllamaError(_error_text(r.status_code, body))
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if obj.get("error"):
                        raise OllamaError(str(obj["error"]))
                    yield obj
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host} ({type(exc).__name__}). "
                "Is `ollama serve` running?") from exc


def _error_text(status: int, body: str) -> str:
    try:
        msg = json.loads(body).get("error", body)
    except (ValueError, AttributeError):
        msg = body
    return f"Ollama error {status}: {str(msg)[:400]}"
