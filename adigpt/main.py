"""FastAPI app: JSON API, Server-Sent Events chat stream, and the static web UI."""
from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, prompts
from .chat import ChatEngine, ChatRequest
from .config import Config
from .db import DB
from .memory import Memory
from .ollama import Ollama, OllamaError
from .rag import SUPPORTED_EXTS, Retriever

STATIC = Path(__file__).resolve().parent.parent / "static"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]", "testserver", "test"}


class ChatBody(BaseModel):
    message: str = Field("", max_length=200_000)
    conversation_id: int | None = None
    mode: str = "general"
    use_docs: bool = False
    use_web: bool = False
    strict: bool = True
    think: bool = False
    incognito: bool = False
    regenerate: bool = False


class SettingsBody(BaseModel):
    chat_model: str | None = None
    embed_model: str | None = None
    num_ctx: int | None = Field(None, ge=1024, le=262144)
    temperature: float | None = Field(None, ge=0, le=2)
    custom_instructions: str | None = Field(None, max_length=4000)
    auto_memory: bool | None = None
    tools_enabled: bool | None = None
    reset_temperature: bool = False


class TextBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class PathBody(BaseModel):
    path: str


class TitleBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class SearchBody(BaseModel):
    query: str
    method: str = "hybrid"


def _has(installed: list[str], model: str) -> bool:
    return model in installed or f"{model}:latest" in installed


def create_app(cfg: Config | None = None,
               ollama_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    cfg = cfg or Config.from_env()
    db = DB(cfg.db_path)
    ollama = Ollama(cfg.ollama_host, transport=ollama_transport)
    retriever = Retriever(cfg, db, ollama)
    memory = Memory(db)
    engine = ChatEngine(cfg, db, ollama, retriever, memory)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await ollama.close()
        db.close()

    app = FastAPI(title="adi-gpt", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None)
    app.state.cfg, app.state.db, app.state.engine = cfg, db, engine
    app.state.retriever, app.state.memory, app.state.ollama = retriever, memory, ollama
    open_bind = cfg.host not in LOCAL_HOSTS

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        """Block cross-site requests and DNS-rebinding when bound to localhost."""
        header = request.headers.get("host") or ""
        if header.startswith("["):
            host = header.split("]")[0] + "]"
        else:
            host = header.rsplit(":", 1)[0]
        if not open_bind and host not in LOCAL_HOSTS:
            return JSONResponse({"detail": "Host not allowed"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and request.method not in ("GET", "HEAD", "OPTIONS"):
            if urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Cross-origin request blocked"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "frame-ancestors 'none'")
        return response

    # ------------------------------------------------------------ status
    @app.get("/api/status")
    async def status():
        s = engine.settings()
        out = {"app_version": __version__, "ollama_host": cfg.ollama_host, "ok": False,
               "chat_model": s["chat_model"], "embed_model": s["embed_model"],
               "documents": db.one("SELECT COUNT(*) c FROM documents")["c"],
               "chunks": db.one("SELECT COUNT(*) c FROM chunks")["c"],
               "memories": db.one("SELECT COUNT(*) c FROM memories")["c"],
               "allowed_dirs": [str(p) for p in cfg.allowed_dirs]}
        try:
            out["ollama_version"] = await ollama.version()
            out["models"] = await ollama.models()
            out["missing"] = [m for m in (s["chat_model"], s["embed_model"])
                              if not _has(out["models"], m)]
            out["capabilities"] = await ollama.capabilities(s["chat_model"])
            out["ok"] = True
        except OllamaError as exc:
            out["error"] = str(exc)
        return out

    @app.get("/api/modes")
    async def modes():
        return [{"id": k, "label": v["label"], "temperature": v["temperature"]}
                for k, v in prompts.MODES.items()]

    @app.get("/api/settings")
    async def get_settings():
        return engine.settings()

    @app.put("/api/settings")
    async def put_settings(body: SettingsBody):
        values = body.model_dump(exclude_none=True, exclude={"reset_temperature"})
        if body.reset_temperature:
            values["temperature"] = None
        db.set_settings(values)
        return engine.settings()

    # ------------------------------------------------------------- chat
    @app.post("/api/chat")
    async def chat(body: ChatBody):
        async def events():
            try:
                async for event in engine.stream(ChatRequest(**body.model_dump())):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except OllamaError as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------------------------------------------------- conversations
    @app.get("/api/conversations")
    async def conversations(q: str = ""):
        return db.list_conversations(q)

    @app.get("/api/conversations/{conv_id}")
    async def conversation(conv_id: int):
        row = db.one("SELECT * FROM conversations WHERE id=?", (conv_id,))
        if not row:
            raise HTTPException(404, "Conversation not found")
        return {**dict(row), "messages": db.messages(conv_id)}

    @app.patch("/api/conversations/{conv_id}")
    async def rename(conv_id: int, body: TitleBody):
        db.run("UPDATE conversations SET title=? WHERE id=?", (body.title.strip(), conv_id))
        return {"ok": True}

    @app.delete("/api/conversations/{conv_id}")
    async def delete_conversation(conv_id: int):
        db.run("DELETE FROM conversations WHERE id=?", (conv_id,))
        return {"ok": True}

    @app.get("/api/conversations/{conv_id}/export")
    async def export(conv_id: int):
        row = db.one("SELECT title FROM conversations WHERE id=?", (conv_id,))
        if not row:
            raise HTTPException(404, "Conversation not found")
        lines = [f"# {row['title']}", ""]
        for m in db.messages(conv_id):
            lines += [f"## {'You' if m['role'] == 'user' else 'adi-gpt'}", "", m["content"], ""]
            for src in m["meta"].get("sources") or []:
                lines.append(f"- [S{src['n']}] {src['title']} {src.get('url') or src.get('path') or ''}")
            if m["meta"].get("sources"):
                lines.append("")
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", row["title"]).strip("-")[:50] or "chat"
        return PlainTextResponse("\n".join(lines), media_type="text/markdown", headers={
            "Content-Disposition": f'attachment; filename="{name}.md"'})

    # -------------------------------------------------------- documents
    @app.get("/api/documents")
    async def documents():
        return [dict(r) for r in db.q(
            "SELECT id, path, title, size, n_chunks, indexed_at FROM documents ORDER BY title")]

    @app.post("/api/documents/upload")
    async def upload(files: list[UploadFile]):
        cfg.upload_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for f in files:
            name = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(f.filename or "file").name).strip(". ")
            if Path(name).suffix.lower() not in SUPPORTED_EXTS:
                results.append({"file": f.filename, "status": "unsupported type"})
                continue
            data = await f.read(cfg.max_file_bytes + 1)
            if len(data) > cfg.max_file_bytes:
                results.append({"file": f.filename, "status": "too large"})
                continue
            target = cfg.upload_dir / name
            target.write_bytes(data)
            try:
                results.append({"file": name, "status": await retriever.index_file(target, name)})
            except OllamaError as exc:
                raise HTTPException(503, str(exc)) from exc
        return results

    @app.post("/api/documents/index-folder")
    async def index_folder(body: PathBody):
        try:
            return await retriever.index_folder(Path(body.path))
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except OllamaError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.delete("/api/documents/{doc_id}")
    async def delete_document(doc_id: int):
        row = db.one("SELECT path FROM documents WHERE id=?", (doc_id,))
        if row:
            retriever.delete_document(doc_id)
            p = Path(row["path"])
            if cfg.upload_dir.resolve() in p.parents:  # only uploaded copies are removed
                p.unlink(missing_ok=True)
        return {"ok": True}

    @app.post("/api/documents/search")
    async def search_documents(body: SearchBody):
        try:
            hits = await retriever.search(body.query, method=body.method)
        except OllamaError as exc:
            raise HTTPException(503, str(exc)) from exc
        return [{**h, "text": h["text"][:600]} for h in hits]

    # ----------------------------------------------------------- memory
    @app.get("/api/memories")
    async def memories():
        return memory.all()

    @app.post("/api/memories")
    async def add_memory(body: TextBody):
        mem_id = memory.add(body.text)
        if mem_id is None:
            raise HTTPException(422, "Not saved: empty, duplicate, or looks like a secret.")
        return {"id": mem_id}

    @app.patch("/api/memories/{mem_id}")
    async def edit_memory(mem_id: int, body: TextBody):
        if not memory.update(mem_id, body.text):
            raise HTTPException(422, "Not saved: empty or looks like a secret.")
        return {"ok": True}

    @app.delete("/api/memories/{mem_id}")
    async def delete_memory(mem_id: int):
        memory.delete(mem_id)
        return {"ok": True}

    @app.delete("/api/memories")
    async def clear_memories():
        memory.clear()
        return {"ok": True}

    # ------------------------------------------------------------ static
    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def run() -> None:
    import uvicorn
    cfg = Config.from_env()
    print(f"adi-gpt → http://{cfg.host}:{cfg.port}  (Ollama: {cfg.ollama_host})")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    run()
