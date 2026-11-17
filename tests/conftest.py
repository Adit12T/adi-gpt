import json

import httpx
import pytest

from adigpt.config import Config
from adigpt.main import create_app
from adigpt.testing import FakeOllama


@pytest.fixture
def fake():
    return FakeOllama()


@pytest.fixture
def cfg(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    return Config(data_dir=tmp_path / "data", allowed_dirs=[docs.resolve()])


@pytest.fixture
def app(cfg, fake):
    return create_app(cfg, ollama_transport=fake.transport)


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture
def docs_dir(cfg):
    return cfg.allowed_dirs[0]


async def chat(client, **body) -> list[dict]:
    """POST /api/chat and return the parsed SSE events."""
    r = await client.post("/api/chat", json=body)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]


def text_of(events: list[dict]) -> str:
    return "".join(e["text"] for e in events if e["type"] == "token")


def one(events: list[dict], kind: str) -> dict:
    found = [e for e in events if e["type"] == kind]
    assert found, f"no {kind} event in {[e['type'] for e in events]}"
    return found[-1]
