"""SQLite storage: one file holds chats, documents, chunks, memories, settings."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY, title TEXT NOT NULL DEFAULT 'New chat',
  mode TEXT NOT NULL DEFAULT 'general', incognito INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY,
  conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  content, content='messages', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content); END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content); END;

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
  sha256 TEXT NOT NULL, size INTEGER NOT NULL, n_chunks INTEGER NOT NULL DEFAULT 0,
  indexed_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ord INTEGER NOT NULL, heading TEXT NOT NULL DEFAULT '', text TEXT NOT NULL,
  embedding BLOB, embed_model TEXT);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, content='chunks', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text); END;

CREATE TABLE IF NOT EXISTS embed_cache (
  hash TEXT PRIMARY KEY, embedding BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'manual',
  created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class DB:
    """Thin, thread-safe wrapper around one SQLite connection (single-user app)."""

    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        for pragma in ("journal_mode=WAL", "synchronous=NORMAL", "foreign_keys=ON",
                       "busy_timeout=5000"):
            self.conn.execute(f"PRAGMA {pragma}")
        self.conn.executescript(SCHEMA)
        self.index_version = 0  # bumped whenever chunks change (vector cache key)

    def close(self) -> None:
        self.conn.close()

    def q(self, sql: str, args: tuple | list = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def one(self, sql: str, args: tuple | list = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(sql, args).fetchone()

    def run(self, sql: str, args: tuple | list = ()) -> int:
        with self.lock:
            return self.conn.execute(sql, args).lastrowid

    def many(self, sql: str, rows: list[tuple]) -> None:
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.executemany(sql, rows)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    # ---- settings -------------------------------------------------------
    def get_settings(self) -> dict[str, Any]:
        return {r["key"]: json.loads(r["value"]) for r in self.q("SELECT key, value FROM settings")}

    def set_settings(self, values: dict[str, Any]) -> None:
        for k, v in values.items():
            self.run("INSERT INTO settings(key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, json.dumps(v)))

    # ---- conversations --------------------------------------------------
    def create_conversation(self, title: str = "New chat", mode: str = "general",
                            incognito: bool = False) -> int:
        now = time.time()
        return self.run("INSERT INTO conversations(title, mode, incognito, created_at, updated_at)"
                        " VALUES (?,?,?,?,?)", (title, mode, int(incognito), now, now))

    def add_message(self, conv_id: int, role: str, content: str, meta: dict | None = None) -> int:
        now = time.time()
        mid = self.run("INSERT INTO messages(conversation_id, role, content, meta, created_at)"
                       " VALUES (?,?,?,?,?)", (conv_id, role, content, json.dumps(meta or {}), now))
        self.run("UPDATE conversations SET updated_at=? WHERE id=?", (now, conv_id))
        return mid

    def messages(self, conv_id: int) -> list[dict]:
        rows = self.q("SELECT id, role, content, meta, created_at FROM messages "
                      "WHERE conversation_id=? ORDER BY id", (conv_id,))
        return [{**dict(r), "meta": json.loads(r["meta"])} for r in rows]

    def list_conversations(self, query: str = "") -> list[dict]:
        if query.strip():
            match = fts_query(query)
            if not match:
                return []
            hits = self.q(
                "SELECT c.id, c.title, c.mode, c.incognito, c.updated_at, "
                "snippet(messages_fts, 0, '', '', '…', 12) AS snippet "
                "FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid "
                "JOIN conversations c ON c.id = m.conversation_id "
                "WHERE messages_fts MATCH ? ORDER BY c.updated_at DESC, rank LIMIT 300",
                (match,))
            seen: set[int] = set()
            rows = [r for r in hits if not (r["id"] in seen or seen.add(r["id"]))][:50]
        else:
            rows = self.q("SELECT id, title, mode, incognito, updated_at, '' AS snippet "
                          "FROM conversations ORDER BY updated_at DESC LIMIT 200")
        return [dict(r) for r in rows]


STOPWORDS = frozenset(
    "a an and are as at be but by do does for from how i in is it my of on or that the "
    "this to was what when where which who why will with you your me we our can".split())


def fts_terms(text: str) -> list[str]:
    import re
    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    terms = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return terms or [w for w in words if len(w) > 1]


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query (quoted terms joined with OR)."""
    seen: list[str] = []
    for t in fts_terms(text):
        if t not in seen:
            seen.append(t)
    return " OR ".join(f'"{t}"' for t in seen[:32])
