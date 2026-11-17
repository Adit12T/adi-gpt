"""Long-term memory: short facts about the user, always visible and editable in the UI."""
from __future__ import annotations

import json
import re
import time

from .db import DB
from .ollama import Ollama, OllamaError

MAX_FACTS_IN_PROMPT = 40
MAX_FACT_CHARS = 300

# Things that must never be stored: keys, tokens, passwords, card-like numbers.
_SECRET = re.compile(
    r"(?i)(password|passwd|secret|api[_ -]?key|token|bearer|private key|ssn|cvv)"
    r"|sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
    r"|\b(?:\d[ -]?){13,19}\b|-----BEGIN")

_REMEMBER = re.compile(r"(?is)^\s*(?:please\s+)?remember(?:\s+that|\s*:)\s+(.+)$")


def looks_secret(text: str) -> bool:
    return bool(_SECRET.search(text))


def explicit_fact(message: str) -> str | None:
    """'remember that …' / 'remember: …' → the fact to store."""
    m = _REMEMBER.match(message)
    return m.group(1).strip() if m else None


class Memory:
    def __init__(self, db: DB):
        self.db = db

    def all(self) -> list[dict]:
        return [dict(r) for r in self.db.q(
            "SELECT id, text, source, created_at FROM memories ORDER BY id")]

    def for_prompt(self) -> list[str]:
        rows = self.db.q("SELECT text FROM memories ORDER BY id DESC LIMIT ?",
                         (MAX_FACTS_IN_PROMPT,))
        return [r["text"] for r in reversed(rows)]

    def add(self, text: str, source: str = "manual") -> int | None:
        text = " ".join(text.split())[:MAX_FACT_CHARS]
        if not text or looks_secret(text):
            return None
        if self.db.one("SELECT 1 FROM memories WHERE lower(text)=lower(?)", (text,)):
            return None
        return self.db.run("INSERT INTO memories(text, source, created_at) VALUES (?,?,?)",
                           (text, source, time.time()))

    def update(self, mem_id: int, text: str) -> bool:
        text = " ".join(text.split())[:MAX_FACT_CHARS]
        if not text or looks_secret(text):
            return False
        self.db.run("UPDATE memories SET text=? WHERE id=?", (text, mem_id))
        return True

    def delete(self, mem_id: int) -> None:
        self.db.run("DELETE FROM memories WHERE id=?", (mem_id,))

    def clear(self) -> None:
        self.db.run("DELETE FROM memories")


EXTRACT_SCHEMA = {"type": "object", "required": ["facts"], "properties": {
    "facts": {"type": "array", "items": {"type": "string"}}}}

EXTRACT_PROMPT = """Extract durable facts about the user from their message below: role, \
tech stack, preferences, ongoing projects, goals. Only facts the user states about \
themselves. Skip questions, one-off task details, secrets and anything uncertain. \
Each fact is one short third-person sentence. Respond in JSON: {"facts": [...]}. \
Return {"facts": []} if there is nothing durable.

Already known:
%s

User message:
%s"""


async def extract_facts(ollama: Ollama, model: str, memory: Memory, message: str,
                        num_ctx: int, keep_alive: str) -> list[str]:
    """Opt-in background step: ask the model for durable facts (structured output)."""
    if len(message) < 20 or looks_secret(message):
        return []
    known = "\n".join(f"- {m}" for m in memory.for_prompt()) or "(nothing)"
    try:
        resp = await ollama.chat({
            "model": model, "format": EXTRACT_SCHEMA, "think": False, "keep_alive": keep_alive,
            "options": {"temperature": 0, "num_ctx": num_ctx},
            "messages": [{"role": "user", "content": EXTRACT_PROMPT % (known, message[:4000])}]})
        facts = json.loads(resp["message"]["content"]).get("facts", [])
    except (OllamaError, ValueError, KeyError, TypeError):
        return []
    saved = []
    for fact in facts[:5]:
        if isinstance(fact, str) and memory.add(fact, source="auto"):
            saved.append(fact)
    return saved
