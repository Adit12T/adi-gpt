# Research notes behind adi-gpt (November 2026)

Design decisions and the sources they rest on. Versions and model tags move quickly;
re-check the linked pages before relying on a specific number.

## Ollama API

- The app uses the native API: `POST /api/chat`, `POST /api/embed`, `POST /api/show`,
  `GET /api/tags`, `GET /api/version`. Streaming is newline-delimited JSON, which the backend
  re-emits to the browser as Server-Sent Events. <https://docs.ollama.com/api/chat>
- Tool calling uses an OpenAI-style `tools` array; results go back as `role: "tool"` messages
  with `tool_name`. <https://docs.ollama.com/capabilities/tool-calling>
- `format` accepts a JSON schema (used for memory extraction); `think` toggles reasoning and
  is only sent to models whose `/api/show` capabilities include `thinking`.
- Context length: Ollama's documented default is small (4096 in the FAQ) and long prompts are
  cut silently, so the app always sends `options.num_ctx`. <https://docs.ollama.com/faq>
- `/api/embed` (plural input, batched) is current; `/api/embeddings` is legacy.
  <https://ollama.com/blog/embedding-models>
- The OpenAI-compatible `/v1` API cannot pass `num_ctx`, so it is not used.
  <https://docs.ollama.com/api/openai-compatibility>

## Models

Tags reported on the Ollama library pages at research time: `qwen3.5:4b`, `qwen3.5:9b`
(default chat model: tools, thinking, vision), `qwen3.6:27b-coding`, `qwen3-coder:30b`,
`nomic-embed-text` (768 dimensions, needs `search_query:` / `search_document:` prefixes,
which the app adds). <https://ollama.com/library/qwen3.5/tags>
<https://ollama.com/library/nomic-embed-text>

## Reducing hallucination

- Chunking: Chroma's evaluation found recursive chunks of 200–400 tokens with little or no
  overlap beat the common 800/400 default on recall and precision. The app uses
  structure-aware chunks of about 400 tokens with no overlap.
  <https://research.trychroma.com/evaluating-chunking>
- Hybrid search: FTS5 BM25 and vector rankings fused with Reciprocal Rank Fusion
  (`1 / (60 + rank)`). Keyword search catches identifiers and error codes that embeddings miss.
- Cite-or-abstain prompting plus a relevance floor, with a programmatic citation check after
  generation. A second "self-check" model pass was left out because it doubles latency locally.
- Low temperature (0.2 at most) for grounded answers.

## Choices that differ from the research report

- **Vectors in NumPy, not sqlite-vec.** sqlite-vec is pre-1.0 and needs SQLite extension
  loading, which some Python builds (notably macOS system Python) disable. Embeddings are
  stored as float32 blobs in SQLite and searched by brute force.
- **httpx instead of the `ollama` client package.** Five endpoints are used; a thin client
  keeps dependencies down and makes the model easy to mock in tests.
- **pypdf instead of PyMuPDF.** PyMuPDF is AGPL-licensed; pypdf is permissive.
- **`StreamingResponse` for SSE**, which works on every FastAPI version in the supported range.

## Not built yet

Reranking, rolling conversation summaries, image input, MCP client support, a SearXNG
provider, and answer-quality scoring with an LLM judge (Ragas / DeepEval).
