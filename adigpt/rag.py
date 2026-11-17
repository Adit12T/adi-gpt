"""Document ingestion and hybrid retrieval (BM25 via FTS5 + embeddings, fused with RRF)."""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
from pathlib import Path

import numpy as np

from .config import Config
from .db import DB, fts_query, fts_terms
from .ollama import Ollama

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".java", ".kt", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yml", ".yaml", ".xml", ".sql", ".properties", ".gradle", ".sh", ".csv", ".log",
    ".go", ".rs", ".c", ".h", ".cpp", ".cs", ".css", ".scss", ".toml", ".ini", ".cfg", ".tex",
}
OTHER_EXTS = {".pdf", ".docx", ".html", ".htm"}
SUPPORTED_EXTS = TEXT_EXTS | OTHER_EXTS
SKIP_DIRS = {".git", "node_modules", "target", "build", "dist", ".idea", ".gradle", ".venv",
             "venv", "__pycache__", ".mvn", "out", ".next", ".cache"}
MAX_CHARS = 1600  # ~400 tokens per chunk
RRF_K = 60


# ---------------------------------------------------------------- parsing
def parse_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        import docx
        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs if p.text.strip())
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if ext in (".html", ".htm"):
        try:
            import trafilatura
            extracted = trafilatura.extract(text)
            if extracted:
                return extracted
        except Exception:
            pass
        return re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(script|style).*?</\1>", " ", text))
    return text


# --------------------------------------------------------------- chunking
def _pack(blocks: list[str], limit: int = MAX_CHARS) -> list[str]:
    """Greedily pack blocks into chunks of at most `limit` characters."""
    out: list[str] = []
    cur = ""
    for block in blocks:
        block = block.strip("\n")
        if not block.strip():
            continue
        while len(block) > limit:  # oversized block: cut at a line or space boundary
            cut = block.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = block.rfind(" ", 0, limit)
            if cut < limit // 2:
                cut = limit
            if cur:
                out.append(cur)
                cur = ""
            out.append(block[:cut])
            block = block[cut:].lstrip()
        if not block:
            continue
        if cur and len(cur) + len(block) + 2 > limit:
            out.append(cur)
            cur = block
        else:
            cur = f"{cur}\n\n{block}" if cur else block
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, path: str = "") -> list[tuple[str, str]]:
    """Split text into (heading, chunk) pairs, following document structure."""
    text = text.replace("\r\n", "\n").replace("\x00", "")
    if not text.strip():
        return []
    sections: list[tuple[str, str]] = []
    if Path(path).suffix.lower() in (".md", ".markdown"):
        trail: list[str] = []
        body: list[str] = []
        in_code = False

        def flush() -> None:
            if "".join(body).strip():
                sections.append((" > ".join(trail), "\n".join(body)))
            body.clear()

        for line in text.split("\n"):
            if line.lstrip().startswith("```"):
                in_code = not in_code
            m = None if in_code else re.match(r"^(#{1,6})\s+(.*\S)", line)
            if m:
                flush()
                level = len(m.group(1))
                del trail[level - 1:]
                trail.append(m.group(2).strip("# ").strip())
            else:
                body.append(line)
        flush()
    else:
        sections.append(("", text))
    chunks: list[tuple[str, str]] = []
    for heading, body_text in sections:
        for piece in _pack(re.split(r"\n\s*\n", body_text)):
            chunks.append((heading, piece))
    return chunks


# -------------------------------------------------------------- embeddings
def _prefix(model: str, kind: str) -> str:
    # nomic-embed-text is trained with task prefixes.
    if "nomic-embed" in model:
        return "search_query: " if kind == "query" else "search_document: "
    return ""


def _blob(vec: list[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


class Retriever:
    def __init__(self, cfg: Config, db: DB, ollama: Ollama):
        self.cfg, self.db, self.ollama = cfg, db, ollama
        self._matrix: tuple[int, str, np.ndarray, np.ndarray] | None = None

    @property
    def embed_model(self) -> str:
        return self.db.get_settings().get("embed_model") or self.cfg.embed_model

    async def embed(self, texts: list[str], kind: str) -> list[bytes]:
        """Embed texts, reusing cached vectors for text seen before."""
        model = self.embed_model
        full = [_prefix(model, kind) + t for t in texts]
        keys = [hashlib.sha256(f"{model}\0{t}".encode()).hexdigest() for t in full]
        found: dict[str, bytes] = {}
        if kind == "document":
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                rows = self.db.q("SELECT hash, embedding FROM embed_cache WHERE hash IN (%s)"
                                 % ",".join("?" * len(part)), part)
                found.update({r["hash"]: r["embedding"] for r in rows})
        todo = [i for i, k in enumerate(keys) if k not in found]
        if todo:
            vecs = await self.ollama.embed(model, [full[i] for i in todo], self.cfg.keep_alive)
            for i, v in zip(todo, vecs):
                found[keys[i]] = _blob(v)
            if kind == "document":
                self.db.many("INSERT OR REPLACE INTO embed_cache(hash, embedding) VALUES (?,?)",
                             [(keys[i], found[keys[i]]) for i in todo])
        return [found[k] for k in keys]

    # ----------------------------------------------------------- indexing
    def is_allowed(self, path: Path) -> bool:
        resolved = path.expanduser().resolve()
        roots = [*self.cfg.allowed_dirs, self.cfg.upload_dir.resolve()]
        return any(resolved == r or r in resolved.parents for r in roots)

    async def index_file(self, path: Path, title: str | None = None) -> str:
        path = path.expanduser().resolve()
        if not self.is_allowed(path):
            raise PermissionError(f"{path} is outside the allowed folders")
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return "skipped"
        size = path.stat().st_size
        if size == 0 or size > self.cfg.max_file_bytes:
            return "skipped"
        digest = await asyncio.to_thread(lambda: hashlib.sha256(path.read_bytes()).hexdigest())
        model = self.embed_model
        row = self.db.one("SELECT id, sha256 FROM documents WHERE path=?", (str(path),))
        if row and row["sha256"] == digest and not self.db.one(
                "SELECT 1 FROM chunks WHERE document_id=? AND embed_model IS NOT ? LIMIT 1",
                (row["id"], model)):
            return "unchanged"
        try:
            text = await asyncio.to_thread(parse_file, path)
        except Exception:
            return "skipped"
        title = title or path.name
        chunks = chunk_text(text, str(path))
        if not chunks:
            return "skipped"
        vecs = await self.embed(
            [f"{title}{' > ' + h if h else ''}\n{c}" for h, c in chunks], "document")
        with self.db.lock:
            if row:
                self.db.run("DELETE FROM documents WHERE id=?", (row["id"],))
            doc_id = self.db.run(
                "INSERT INTO documents(path, title, sha256, size, n_chunks, indexed_at) "
                "VALUES (?,?,?,?,?,?)", (str(path), title, digest, size, len(chunks), time.time()))
            self.db.many(
                "INSERT INTO chunks(document_id, ord, heading, text, embedding, embed_model) "
                "VALUES (?,?,?,?,?,?)",
                [(doc_id, i, h, c, v, model) for i, ((h, c), v) in enumerate(zip(chunks, vecs))])
            self.db.index_version += 1
        return "indexed"

    async def index_folder(self, folder: Path) -> dict:
        folder = folder.expanduser().resolve()
        if not self.is_allowed(folder):
            raise PermissionError(f"{folder} is outside the allowed folders")
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} is not a folder")
        counts = {"indexed": 0, "unchanged": 0, "skipped": 0, "removed": 0}
        seen: set[str] = set()
        stack = [folder]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(current.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if entry.name not in SKIP_DIRS and not entry.name.startswith("."):
                        stack.append(entry)
                elif entry.suffix.lower() in SUPPORTED_EXTS:
                    seen.add(str(entry))
                    try:
                        counts[await self.index_file(entry)] += 1
                    except OSError:
                        counts["skipped"] += 1
        # Drop documents under this folder whose files no longer exist.
        prefix = str(folder).rstrip("/\\")
        for row in self.db.q("SELECT id, path FROM documents"):
            p = row["path"]
            if (p.startswith(prefix + "/") or p.startswith(prefix + "\\")) and p not in seen \
                    and not Path(p).exists():
                self.delete_document(row["id"])
                counts["removed"] += 1
        return counts

    def delete_document(self, doc_id: int) -> None:
        with self.db.lock:
            self.db.run("DELETE FROM documents WHERE id=?", (doc_id,))
            self.db.index_version += 1

    # ------------------------------------------------------------- search
    def _vectors(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._matrix
        if cached and cached[0] == self.db.index_version and cached[1] == model:
            return cached[2], cached[3]
        rows = self.db.q("SELECT id, embedding FROM chunks WHERE embed_model=? "
                         "AND embedding IS NOT NULL", (model,))
        ids = np.array([r["id"] for r in rows], dtype=np.int64)
        if rows:
            mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
        else:
            mat = np.zeros((0, 1), dtype=np.float32)
        self._matrix = (self.db.index_version, model, ids, mat)
        return ids, mat

    def keyword_ranking(self, query: str, limit: int = 30) -> list[int]:
        match = fts_query(query)
        if not match:
            return []
        rows = self.db.q("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                         "ORDER BY bm25(chunks_fts) LIMIT ?", (match, limit))
        return [r["rowid"] for r in rows]

    async def vector_ranking(self, query: str, limit: int = 30) -> list[tuple[int, float]]:
        model = self.embed_model
        ids, mat = self._vectors(model)
        if len(ids) == 0:
            return []
        qv = np.frombuffer((await self.embed([query], "query"))[0], dtype=np.float32)
        if qv.shape[0] != mat.shape[1]:
            return []
        qv = qv / max(float(np.linalg.norm(qv)), 1e-12)
        sims = mat @ qv
        top = np.argsort(-sims)[:limit]
        return [(int(ids[i]), float(sims[i])) for i in top]

    async def search(self, query: str, k: int | None = None, method: str = "hybrid") -> list[dict]:
        """Return the top-k chunks. Each hit carries `relevant`, used for abstention."""
        k = k or self.cfg.top_k
        kw = self.keyword_ranking(query) if method in ("hybrid", "bm25") else []
        vec = await self.vector_ranking(query) if method in ("hybrid", "vector") else []
        sims = dict(vec)
        scores: dict[int, float] = {}
        for ranking in (kw, [cid for cid, _ in vec]):
            for rank, cid in enumerate(ranking):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(scores, key=lambda c: -scores[c])[:k]
        if not best:
            return []
        rows = self.db.q(
            "SELECT c.id, c.heading, c.text, d.id AS document_id, d.title, d.path "
            "FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id IN (%s)"
            % ",".join("?" * len(best)), best)
        by_id = {r["id"]: dict(r) for r in rows}
        terms = set(fts_terms(query))
        hits = []
        for cid in best:
            hit = by_id.get(cid)
            if not hit:
                continue
            words = set(fts_terms(hit["text"] + " " + hit["heading"]))
            overlap = len(terms & words) / len(terms) if terms else 0.0
            sim = sims.get(cid)
            hit.update(score=round(scores[cid], 5), similarity=None if sim is None else round(sim, 4),
                       keyword_overlap=round(overlap, 2), kind="doc",
                       relevant=bool((sim is not None and sim >= self.cfg.min_similarity)
                                     or overlap >= 0.5))
            hits.append(hit)
        return hits


# -------------------------------------------------------------- citations
ABSTAIN = "NOT_IN_SOURCES"
_CITE = re.compile(r"\[S(\d+)\]")


def format_sources(sources: list[dict]) -> str:
    parts = []
    for i, s in enumerate(sources, 1):
        label = s["title"] + (f" — {s['heading']}" if s.get("heading") else "")
        if s.get("url"):
            label += f" ({s['url']})"
        parts.append(f"[S{i}] {label}\n{s['text']}")
    return "<sources>\n" + "\n\n".join(parts) + "\n</sources>"


def verify_citations(answer: str, sources: list[dict]) -> dict:
    """Programmatic check of [S#] citations: no model call involved."""
    n = len(sources)
    cited = sorted({int(m) for m in _CITE.findall(answer)})
    invalid = [c for c in cited if c < 1 or c > n]
    prose = re.sub(r"```.*?```", "", answer, flags=re.S)
    checked = uncited = 0
    weak: set[int] = set()
    for para in re.split(r"\n\s*\n|\n(?=\s*(?:[-*]|\d+\.)\s)", prose):
        para = para.strip()
        if len(para.split()) < 8 or para.startswith("#"):
            continue
        checked += 1
        refs = [int(m) for m in _CITE.findall(para)]
        if not refs:
            uncited += 1
            continue
        terms = set(fts_terms(_CITE.sub("", para)))
        for ref in refs:
            if 1 <= ref <= n and terms:
                src_terms = set(fts_terms(sources[ref - 1]["text"]))
                if len(terms & src_terms) / len(terms) < 0.2:
                    weak.add(ref)
    return {"cited": [c for c in cited if c not in invalid], "invalid": invalid,
            "weak": sorted(weak), "checked": checked, "uncited": uncited,
            "abstained": ABSTAIN in answer, "ok": not invalid and uncited == 0 and not weak}
