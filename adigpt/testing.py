"""A fake Ollama server (httpx transport) for tests, CI and offline evaluation.

It implements the same endpoints the app uses. Embeddings are deterministic
bag-of-words hashes, so retrieval behaves sensibly without a real model.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import httpx

DIM = 384


def hash_embedding(text: str) -> list[float]:
    text = re.sub(r"^search_(query|document): ", "", text)
    vec = [0.0] * DIM
    for word in re.findall(r"[a-z0-9_]+", text.lower()):
        if len(word) < 3:
            continue
        h = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "little")
        vec[h % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeOllama:
    def __init__(self):
        self.calls: list[dict] = []        # every /api/chat payload received
        self.embed_calls = 0
        self.embedded_texts = 0
        self.down = False
        self.capabilities = ["completion", "tools", "thinking"]
        self.models = ["qwen3.5:9b", "nomic-embed-text:latest"]
        self.reply: str | None = None      # force the next answers
        self.facts = ["The user leads a Java team."]

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.1"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if path == "/api/show":
            return httpx.Response(200, json={"capabilities": self.capabilities})
        if path == "/api/embed":
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            self.embed_calls += 1
            self.embedded_texts += len(inputs)
            return httpx.Response(200, json={"embeddings": [hash_embedding(t) for t in inputs]})
        if path == "/api/chat":
            self.calls.append(body)
            return self._chat(body)
        return httpx.Response(404, json={"error": "not found"})

    def _chat(self, body: dict) -> httpx.Response:
        last = body["messages"][-1]
        if not body.get("stream"):
            content = json.dumps({"facts": self.facts}) if body.get("format") else "ok"
            return httpx.Response(200, json={"message": {"role": "assistant", "content": content},
                                             "done": True})
        tool_calls = None
        if self.reply is not None:
            text = self.reply
        elif last["role"] == "tool":
            text = f"The result is {last['content']}."
        elif body.get("tools") and "calculate" in last["content"].lower():
            expr = re.search(r"calculate\s+([0-9+\-*/(). ]+)", last["content"], re.I)
            text = ""
            tool_calls = [{"function": {"name": "calculator",
                                        "arguments": {"expression": expr.group(1).strip()}}}]
        elif "<sources>" in last["content"]:
            block = last["content"].rsplit("<sources>\n", 1)[1]
            first = re.search(r"\[S1\] [^\n]*\n([^\n]+)", block)
            text = f"{first.group(1)[:200]} [S1]"
        else:
            text = "Hello from the fake model. This is a streamed reply."
        lines = []
        if body.get("think"):
            lines.append({"message": {"role": "assistant", "content": "", "thinking": "hmm "},
                          "done": False})
        for piece in re.findall(r"\S+\s*", text):
            lines.append({"message": {"role": "assistant", "content": piece}, "done": False})
        final = {"message": {"role": "assistant", "content": ""}, "done": True,
                 "done_reason": "stop", "prompt_eval_count": 42, "eval_count": 10,
                 "eval_duration": 500_000_000, "load_duration": 1_000_000}
        if tool_calls:
            final["message"]["tool_calls"] = tool_calls
        lines.append(final)
        data = "".join(json.dumps(line) + "\n" for line in lines).encode()
        return httpx.Response(200, content=data, headers={"content-type": "application/x-ndjson"})
