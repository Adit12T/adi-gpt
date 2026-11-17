"""Chat orchestration: memory, retrieval, tool loop, streaming, citation checks."""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from . import prompts
from .config import Config
from .db import DB
from .memory import Memory, explicit_fact, extract_facts, looks_secret
from .ollama import Ollama, OllamaError
from .rag import ABSTAIN, Retriever, format_sources, verify_citations
from .tools import TOOL_SPECS, run_tool

CHARS_PER_TOKEN = 3.5  # rough, conservative estimate for budgeting
ANSWER_RESERVE_TOKENS = 1500
MAX_TOOL_ROUNDS = 4
NO_SOURCES_TEXT = (
    f"{ABSTAIN}\n\nI couldn't find anything relevant in the selected sources, so I'm not "
    "answering from guesswork. Try rephrasing, index the document that covers this, or turn "
    "off **Strict** to let me answer from general knowledge.")


@dataclass
class ChatRequest:
    message: str = ""
    conversation_id: int | None = None
    mode: str = "general"
    use_docs: bool = False
    use_web: bool = False
    strict: bool = True
    think: bool = False
    incognito: bool = False
    regenerate: bool = False


class ChatEngine:
    def __init__(self, cfg: Config, db: DB, ollama: Ollama, retriever: Retriever, memory: Memory):
        self.cfg, self.db, self.ollama = cfg, db, ollama
        self.retriever, self.memory = retriever, memory
        self.web_search = None  # injected lazily; replaceable in tests
        self._background: set[asyncio.Task] = set()

    def settings(self) -> dict:
        s = self.db.get_settings()
        return {
            "chat_model": s.get("chat_model") or self.cfg.chat_model,
            "embed_model": s.get("embed_model") or self.cfg.embed_model,
            "num_ctx": int(s.get("num_ctx") or self.cfg.num_ctx),
            "temperature": s.get("temperature"),  # None = use the mode's default
            "custom_instructions": s.get("custom_instructions", ""),
            "auto_memory": bool(s.get("auto_memory", False)),
            "tools_enabled": bool(s.get("tools_enabled", True)),
        }

    # ------------------------------------------------------------------
    async def stream(self, req: ChatRequest) -> AsyncIterator[dict]:
        s = self.settings()
        message = req.message.strip()
        conv_id = req.conversation_id
        if conv_id is not None and not self.db.one(
                "SELECT 1 FROM conversations WHERE id=?", (conv_id,)):
            yield {"type": "error", "message": "Conversation not found."}
            return

        if req.regenerate:
            last_user = self.db.one(
                "SELECT id, content FROM messages WHERE conversation_id=? AND role='user' "
                "ORDER BY id DESC LIMIT 1", (conv_id,)) if conv_id else None
            if not last_user:
                yield {"type": "error", "message": "Nothing to regenerate."}
                return
            message = last_user["content"]
            self.db.run("DELETE FROM messages WHERE conversation_id=? AND id>?",
                        (conv_id, last_user["id"]))
        else:
            if not message:
                yield {"type": "error", "message": "Empty message."}
                return
            if conv_id is None:
                title = " ".join(message.split())[:60] or "New chat"
                conv_id = self.db.create_conversation(title, req.mode, req.incognito)
            self.db.add_message(conv_id, "user", message)
        conv = self.db.one("SELECT title, incognito FROM conversations WHERE id=?", (conv_id,))
        incognito = bool(conv["incognito"]) or req.incognito
        yield {"type": "start", "conversation_id": conv_id, "title": conv["title"]}

        # "remember that ..." is handled deterministically, without a model call.
        fact = None if req.regenerate else explicit_fact(message)
        if fact:
            if incognito:
                text = "This chat has memory turned off, so I didn't save that."
            elif looks_secret(fact):
                text = "I didn't save that: it looks like a secret, and I never store those."
            elif self.memory.add(fact, source="chat"):
                text = f"Saved to memory: “{fact[:300]}”. You can edit or delete it under Memory."
            else:
                text = "That's already in memory."
            async for ev in self._finish_static(conv_id, text, {"memory": True}):
                yield ev
            return

        # ---- gather sources (documents and/or web) ---------------------
        sources: list[dict] = []
        grounded = req.use_docs or req.use_web
        if req.use_docs:
            try:
                hits = await self.retriever.search(message)
                sources += [h for h in hits if h["relevant"]]
            except OllamaError as exc:
                yield {"type": "warning", "message": f"Document search failed: {exc}"}
        if req.use_web:
            try:
                search = self.web_search
                if search is None:
                    from .web import web_search as search
                sources += await search(message)
            except Exception as exc:  # noqa: BLE001 - search providers fail in many ways
                yield {"type": "warning", "message": f"Web search failed: {type(exc).__name__}"}
        sources = self._fit_sources(sources, s["num_ctx"])
        public = [{"n": i, "title": x["title"], "heading": x.get("heading", ""),
                   "url": x.get("url"), "path": x.get("path"), "kind": x.get("kind", "doc"),
                   "similarity": x.get("similarity"), "preview": x["text"][:400]}
                  for i, x in enumerate(sources, 1)]
        if grounded:
            yield {"type": "sources", "sources": public}
        if grounded and not sources and req.strict:
            meta = {"sources": [], "verify": {"abstained": True, "ok": True}, "no_model": True}
            async for ev in self._finish_static(conv_id, NO_SOURCES_TEXT, meta):
                yield ev
            return

        # ---- build the prompt (stable prefix first, volatile parts last) ----
        memories = [] if incognito else self.memory.for_prompt()
        system = prompts.system_prompt(req.mode, memories, s["custom_instructions"])
        user_content = message
        if sources:
            user_content += ("\n\n" + prompts.grounding_rules(req.strict) + "\n\n"
                             + format_sources(sources))
        elif grounded:
            user_content += ("\n\n(No relevant sources were found. Say so in one line, then "
                             "answer from general knowledge and flag any uncertainty.)")
        history = [m for m in self.db.messages(conv_id)][:-1]
        budget = int((s["num_ctx"] - ANSWER_RESERVE_TOKENS) * CHARS_PER_TOKEN)
        budget -= len(system) + len(user_content)
        kept: list[dict] = []
        for m in reversed(history):
            if m["role"] not in ("user", "assistant") or not m["content"]:
                continue
            budget -= len(m["content"])
            if budget < 0:
                break
            kept.append({"role": m["role"], "content": m["content"]})
        messages = [{"role": "system", "content": system}, *reversed(kept),
                    {"role": "user", "content": user_content}]

        model = s["chat_model"]
        caps = await self.ollama.capabilities(model)
        mode_temp = prompts.MODES.get(req.mode, prompts.MODES["general"])["temperature"]
        temperature = s["temperature"] if s["temperature"] is not None else mode_temp
        if sources:
            temperature = min(float(temperature), 0.2)
        payload: dict = {"model": model, "keep_alive": self.cfg.keep_alive,
                         "options": {"temperature": temperature, "num_ctx": s["num_ctx"]}}
        if "thinking" in caps:
            payload["think"] = bool(req.think)
        if s["tools_enabled"] and "tools" in caps:
            payload["tools"] = TOOL_SPECS

        # ---- generate, running tool calls between rounds --------------
        answer, thinking, tool_log = "", "", []
        stats: dict = {}
        started = time.perf_counter()
        first_token: float | None = None
        saved = False
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                round_text, calls = "", []
                async for chunk in self.ollama.chat_stream({**payload, "messages": messages}):
                    msg = chunk.get("message") or {}
                    if msg.get("thinking"):
                        thinking += msg["thinking"]
                        yield {"type": "thinking", "text": msg["thinking"]}
                    if msg.get("content"):
                        if first_token is None:
                            first_token = time.perf_counter() - started
                        round_text += msg["content"]
                        yield {"type": "token", "text": msg["content"]}
                    calls += msg.get("tool_calls") or []
                    if chunk.get("done"):
                        stats = _stats(chunk, stats)
                answer += round_text
                if not calls:
                    break
                messages.append({"role": "assistant", "content": round_text, "tool_calls": calls})
                for call in calls:
                    fn = call.get("function") or {}
                    name, args = fn.get("name", ""), fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    result = run_tool(name, args)
                    tool_log.append({"name": name, "arguments": args, "result": result})
                    yield {"type": "tool", "name": name, "arguments": args, "result": result}
                    messages.append({"role": "tool", "tool_name": name, "content": result})
            else:
                yield {"type": "warning", "message": "Stopped after too many tool calls."}
        except OllamaError as exc:
            yield {"type": "error", "message": str(exc)}
            if not answer:
                return
        except (asyncio.CancelledError, GeneratorExit):
            if answer:  # user pressed Stop: keep what was generated
                self.db.add_message(conv_id, "assistant", answer, {"stopped": True, "model": model})
            saved = True
            raise

        stats["seconds"] = round(time.perf_counter() - started, 2)
        if first_token is not None:
            stats["first_token_seconds"] = round(first_token, 2)
        if stats.get("prompt_tokens", 0) >= 0.95 * s["num_ctx"]:
            stats["context_full"] = True
        meta: dict = {"model": model, "stats": stats}
        if tool_log:
            meta["tools"] = tool_log
        if thinking:
            meta["thinking"] = thinking
        if grounded:
            meta["sources"] = public
        if sources:
            meta["verify"] = verify_citations(answer, sources)
            yield {"type": "verify", **meta["verify"]}
        if not saved:
            mid = self.db.add_message(conv_id, "assistant", answer, meta)
            yield {"type": "done", "message_id": mid, "conversation_id": conv_id,
                   "stats": stats, "model": model}

        if s["auto_memory"] and not incognito and not req.regenerate:
            small = self.cfg.small_model or model
            task = asyncio.create_task(extract_facts(
                self.ollama, small, self.memory, message, s["num_ctx"], self.cfg.keep_alive))
            self._background.add(task)
            task.add_done_callback(self._background.discard)

    async def _finish_static(self, conv_id: int, text: str, meta: dict) -> AsyncIterator[dict]:
        yield {"type": "token", "text": text}
        mid = self.db.add_message(conv_id, "assistant", text, meta)
        if "verify" in meta:
            yield {"type": "verify", **meta["verify"]}
        yield {"type": "done", "message_id": mid, "conversation_id": conv_id, "stats": {},
               "model": None}

    def _fit_sources(self, sources: list[dict], num_ctx: int) -> list[dict]:
        """Keep sources within roughly half of the context window."""
        budget = int(num_ctx * CHARS_PER_TOKEN * 0.5)
        out = []
        for src in sources:
            text = src["text"][:3000]
            if len(text) > budget and out:
                break
            out.append({**src, "text": text[:budget]})
            budget -= len(text)
        return out


def _stats(chunk: dict, prev: dict) -> dict:
    out = dict(prev)
    out["prompt_tokens"] = chunk.get("prompt_eval_count", 0) or 0
    out["output_tokens"] = prev.get("output_tokens", 0) + (chunk.get("eval_count", 0) or 0)
    dur = chunk.get("eval_duration") or 0
    if dur and chunk.get("eval_count"):
        out["tokens_per_second"] = round(chunk["eval_count"] / dur * 1e9, 1)
    if chunk.get("load_duration"):
        out["load_seconds"] = round(chunk["load_duration"] / 1e9, 2)
    return out
