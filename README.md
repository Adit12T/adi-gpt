# adi-gpt

A private personal assistant that runs on your own machine. A FastAPI backend talks to
[Ollama](https://ollama.com), everything is stored in one SQLite file, and the chat UI is a
plain web page with no build step. Nothing leaves your computer unless you switch on **Web**
for a message.

## What it does

| Feature | Details |
|---|---|
| Streaming chat | Token-by-token replies, Stop, Regenerate, copy buttons, Markdown and code highlighting |
| Cited answers from your files | Index folders or upload files (Markdown, text, code, PDF, DOCX, HTML). Answers cite `[S1]`-style sources you can open |
| Fewer made-up answers | Hybrid search (BM25 + embeddings, fused with RRF), a strict "answer only from sources" mode, an automatic citation check on every grounded answer, and a refusal (`NOT_IN_SOURCES`) with no model call when nothing relevant is found |
| Memory | "remember that …" saves a fact. View, edit and delete facts in the UI. Optional automatic fact extraction. Secrets are never stored. Per-chat **No memory** switch |
| Modes | General, Code review, Java / Spring design, Write tests, Debug, Interview drill, Notes to actions, Explain simply |
| Tools | Exact calculator and current date/time through Ollama tool calling (no `eval`, no code execution) |
| Web search | Optional per message, through `ddgs`; pages are cleaned and cited like documents |
| Chat history | Full-text search, rename, delete, export to Markdown |
| Speed | Stable prompt prefix so Ollama reuses its cache, model kept warm (`keep_alive`), embedding cache, incremental re-indexing, history trimmed to the context window, tokens/s shown per reply |
| Safety | Binds to `127.0.0.1`, blocks cross-site and foreign-host requests, folder allowlist, sanitised Markdown, strict Content-Security-Policy, sources treated as untrusted text |

## Quick start

Requires Python 3.10+ and Ollama.

```bash
# 1. Models (pick sizes that fit your RAM; see below)
ollama pull qwen3.5:9b
ollama pull nomic-embed-text

# 2. App
git clone https://github.com/Adit12T/adi-gpt && cd adi-gpt
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
adi-gpt                                                # or: python -m adigpt.main
```

Open <http://127.0.0.1:8000>. The dot next to the name turns green when Ollama and both
models are found; otherwise a banner shows the exact `ollama pull` command to run.

### Choosing models

The defaults come from the research in [docs/RESEARCH.md](docs/RESEARCH.md) (November 2026).
Model tags change often, so confirm a tag exists on `ollama.com/library` before pulling.

| RAM | Chat model (`ADIGPT_CHAT_MODEL`) |
|---|---|
| 8 GB | `qwen3.5:4b` |
| 16 GB | `qwen3.5:9b` (default) |
| 32 GB+ | `qwen3.6:27b-coding` or `qwen3-coder:30b` for coding |

Embeddings: `nomic-embed-text` (default). If you change the embedding model, re-index your
folders; old chunks are re-embedded automatically.

## Configuration

Copy `.env.example` to `.env`, or set environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama address |
| `ADIGPT_CHAT_MODEL` | `qwen3.5:9b` | Chat model (also changeable in Settings) |
| `ADIGPT_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `ADIGPT_SMALL_MODEL` | *(chat model)* | Model for automatic memory extraction |
| `ADIGPT_NUM_CTX` | `8192` | Context window sent on every request. Ollama's own default can be as low as 4K and silently cuts long prompts |
| `ADIGPT_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded |
| `ADIGPT_ALLOWED_DIRS` | your home folder | Folders that may be indexed, separated by `:` (`;` on Windows) |
| `ADIGPT_MIN_SIMILARITY` | `0.25` | Relevance floor for retrieved passages; tune it with the evaluation below |
| `ADIGPT_TOP_K` | `6` | Passages given to the model |
| `ADIGPT_DATA_DIR` | `./data` | SQLite database and uploaded files |
| `ADIGPT_HOST` / `ADIGPT_PORT` | `127.0.0.1` / `8000` | Where the app listens. There is no login, so keep it on localhost |

## Using it

- **Docs** on: the question is searched in your indexed files and the answer must cite them.
  With **Strict** on, it refuses rather than guesses when the files don't cover the question.
  With Strict off, it answers from general knowledge and says so.
- **Web** on: the question is sent to a search engine (it leaves your machine) and the
  results are cited the same way.
- Under each grounded answer a badge reports the citation check: green when every paragraph
  cites a real source that shares its content, amber or red with the reason otherwise.
- Documents → "Test retrieval" shows which passages a question would pull in, with scores.
- The check is lexical. It catches missing, invented and off-topic citations; it cannot
  prove a sentence is true. Open the cited source for anything that matters.

## Evaluation

```bash
pip install -e ".[dev]"
pytest                               # 36 tests, Ollama is mocked, runs in under a second
python eval/run_eval.py --offline    # pipeline check without Ollama
python eval/run_eval.py              # full run against your real models
```

The golden set (`eval/golden.jsonl`) has 25 answerable and 8 unanswerable questions over a
small fictional corpus (`eval/corpus/`), so a model cannot answer from prior knowledge.

Results recorded when this repository was created (offline mode, fake hash embeddings):

| Search | hit@1 | hit@3 | hit@6 | MRR |
|---|---|---|---|---|
| BM25 only | 88% | 96% | 96% | 0.92 |
| Vector only (fake embeddings) | 84% | 96% | 96% | 0.89 |
| Hybrid | 88% | 100% | 100% | 0.93 |

**Not yet measured:** answer correctness, abstention rate, tool-calling accuracy and speed
with a real model. The build environment had no access to Ollama's model registry, so the
live run has to happen on your machine: `python eval/run_eval.py` prints those numbers and
saves them to `eval/results-live.json`. The offline numbers test the pipeline, not the
quality of a real embedding model.

## How it works

```
browser ── fetch + SSE ──> FastAPI (adigpt/main.py)
                             ├─ chat.py     prompt assembly, tool loop, streaming, citation check
                             ├─ rag.py      parsing, chunking (~400 tokens), FTS5 + vectors, RRF
                             ├─ memory.py   user facts, secret filter, optional extraction
                             ├─ tools.py    calculator, date/time
                             ├─ web.py      ddgs search + trafilatura page cleaning
                             ├─ ollama.py   /api/chat (NDJSON stream), /api/embed, /api/show
                             └─ db.py       SQLite: WAL, FTS5, embeddings as float32 blobs
```

Vectors are searched by brute force with NumPy, which is fast for a personal corpus (tens of
thousands of chunks) and needs no extension. Past roughly 100k chunks, swap in a vector index.

## Limits

- Single user, no authentication. Do not expose the port to a network.
- Scanned PDFs need OCR first; only embedded text is read.
- Files over 5 MB are skipped.
- Long conversations drop their oldest turns once the context budget is full (no summaries yet).
- Web search depends on `ddgs`, which scrapes public search engines and can be rate limited.
