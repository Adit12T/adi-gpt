"""Configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Tiny .env loader: KEY=VALUE lines, existing environment wins."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _dirs(value: str) -> list[Path]:
    return [Path(p).expanduser().resolve() for p in value.split(os.pathsep) if p.strip()]


@dataclass
class Config:
    ollama_host: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3.5:9b"
    embed_model: str = "nomic-embed-text"
    # Model for background memory extraction. Empty = reuse the chat model,
    # which avoids loading a second model into RAM.
    small_model: str = ""
    num_ctx: int = 8192
    keep_alive: str = "30m"
    data_dir: Path = Path("data")
    # Folders the app may index. Defaults to the home directory.
    allowed_dirs: list[Path] = field(default_factory=lambda: [Path.home().resolve()])
    host: str = "127.0.0.1"
    port: int = 8000
    # Below this best cosine similarity (and with no keyword match) retrieved
    # passages are treated as irrelevant. Tune with eval/run_eval.py.
    min_similarity: float = 0.25
    top_k: int = 6
    max_file_bytes: int = 5_000_000

    @property
    def db_path(self) -> Path:
        return self.data_dir / "adigpt.db"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @classmethod
    def from_env(cls) -> Config:
        load_dotenv()
        e = os.environ.get
        cfg = cls(
            ollama_host=e("OLLAMA_HOST", cls.ollama_host).rstrip("/"),
            chat_model=e("ADIGPT_CHAT_MODEL", cls.chat_model),
            embed_model=e("ADIGPT_EMBED_MODEL", cls.embed_model),
            small_model=e("ADIGPT_SMALL_MODEL", ""),
            num_ctx=int(e("ADIGPT_NUM_CTX", cls.num_ctx)),
            keep_alive=e("ADIGPT_KEEP_ALIVE", cls.keep_alive),
            data_dir=Path(e("ADIGPT_DATA_DIR", "data")).expanduser(),
            host=e("ADIGPT_HOST", cls.host),
            port=int(e("ADIGPT_PORT", cls.port)),
            min_similarity=float(e("ADIGPT_MIN_SIMILARITY", cls.min_similarity)),
            top_k=int(e("ADIGPT_TOP_K", cls.top_k)),
        )
        if not cfg.ollama_host.startswith("http"):
            cfg.ollama_host = "http://" + cfg.ollama_host
        if e("ADIGPT_ALLOWED_DIRS"):
            cfg.allowed_dirs = _dirs(e("ADIGPT_ALLOWED_DIRS"))
        return cfg
