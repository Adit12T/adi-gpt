"""Evaluate adi-gpt: retrieval quality, abstention, grounded answers, tool calls, speed.

    python eval/run_eval.py            # live: needs Ollama running with your models
    python eval/run_eval.py --offline  # no Ollama: fake embeddings, retrieval checks only

Offline numbers use bag-of-words hash embeddings, so they test the pipeline (chunking,
BM25, fusion, the abstention gate), not the quality of a real embedding or chat model.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from adigpt.chat import ChatEngine, ChatRequest  # noqa: E402
from adigpt.config import Config  # noqa: E402
from adigpt.db import DB  # noqa: E402
from adigpt.memory import Memory  # noqa: E402
from adigpt.ollama import Ollama  # noqa: E402
from adigpt.rag import ABSTAIN, Retriever  # noqa: E402

TOOL_CASES = [("What is 4817 * 362? Use the calculator.", "1743754"),
              ("Calculate (125000 * 0.0725) / 12 exactly.", "755.2"),
              ("What is 2 to the power of 20?", "1048576"),
              ("What is 987654 minus 123456?", "864198"),
              ("Calculate 15% of 2480.", "372")]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


async def ask(engine: ChatEngine, **kw) -> dict:
    out = {"text": "", "tools": [], "verify": None, "sources": [], "stats": {}, "error": None}
    async for ev in engine.stream(ChatRequest(**kw)):
        if ev["type"] == "token":
            out["text"] += ev["text"]
        elif ev["type"] == "tool":
            out["tools"].append(ev)
        elif ev["type"] == "verify":
            out["verify"] = ev
        elif ev["type"] == "sources":
            out["sources"] = ev["sources"]
        elif ev["type"] == "done":
            out["stats"] = ev["stats"]
        elif ev["type"] == "error":
            out["error"] = ev["message"]
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true", help="use the built-in fake Ollama")
    ap.add_argument("--limit", type=int, default=0, help="only the first N golden questions")
    args = ap.parse_args()

    golden = [json.loads(line) for line in (ROOT / "golden.jsonl").read_text().splitlines() if line]
    if args.limit:
        golden = golden[:args.limit]
    answerable = [g for g in golden if g["doc"]]
    unanswerable = [g for g in golden if not g["doc"]]

    base = Config.from_env()
    tmp = tempfile.TemporaryDirectory()
    cfg = Config(**{**base.__dict__, "data_dir": Path(tmp.name),
                    "allowed_dirs": [(ROOT / "corpus").resolve()]})
    transport = None
    if args.offline:
        from adigpt.testing import FakeOllama
        transport = FakeOllama().transport
    db, ollama = DB(cfg.db_path), Ollama(cfg.ollama_host, transport=transport)
    retriever = Retriever(cfg, db, ollama)
    engine = ChatEngine(cfg, db, ollama, retriever, Memory(db))
    report: dict = {"mode": "offline" if args.offline else "live", "embed_model": cfg.embed_model,
                    "chat_model": None if args.offline else cfg.chat_model}

    # 1. indexing ---------------------------------------------------------
    t = time.perf_counter()
    first = await retriever.index_folder(ROOT / "corpus")
    index_s = time.perf_counter() - t
    t = time.perf_counter()
    second = await retriever.index_folder(ROOT / "corpus")
    report["indexing"] = {"files": first["indexed"],
                          "chunks": db.one("SELECT COUNT(*) c FROM chunks")["c"],
                          "seconds": round(index_s, 2), "reindex_seconds": round(time.perf_counter() - t, 3),
                          "reindex_unchanged": second["unchanged"]}

    # 2. retrieval --------------------------------------------------------
    report["retrieval"] = {}
    for method in ("bm25", "vector", "hybrid"):
        ranks = []
        for g in answerable:
            hits = await retriever.search(g["q"], k=6, method=method)
            rank = next((i for i, h in enumerate(hits, 1) if h["title"] == g["doc"]), None)
            ranks.append(rank)
        n = len(ranks)
        report["retrieval"][method] = {
            "hit@1": sum(r == 1 for r in ranks) / n, "hit@3": sum(bool(r) and r <= 3 for r in ranks) / n,
            "hit@6": sum(bool(r) for r in ranks) / n, "mrr": sum(1 / r for r in ranks if r) / n}

    # 3. abstention gate (no model involved) --------------------------------
    async def gated(q: str) -> bool:
        return not any(h["relevant"] for h in await retriever.search(q))
    report["gate"] = {
        "unanswerable_blocked": sum([await gated(g["q"]) for g in unanswerable]) / max(len(unanswerable), 1),
        "answerable_wrongly_blocked": sum([await gated(g["q"]) for g in answerable]) / len(answerable)}

    # 4 + 5. answers and tools (live model only) ----------------------------
    if not args.offline:
        rows, speeds, firsts = [], [], []
        for g in golden:
            r = await ask(engine, message=g["q"], use_docs=True, strict=True)
            if r["error"]:
                print("Model error:", r["error"], file=sys.stderr)
                return 1
            abstained = ABSTAIN in r["text"]
            v = r["verify"] or {}
            cited_titles = {r["sources"][c - 1]["title"] for c in v.get("cited", [])
                            if c <= len(r["sources"])}
            rows.append({"q": g["q"], "answerable": bool(g["doc"]), "abstained": abstained,
                         "correct": bool(g["doc"]) and not abstained and all(
                             a.lower() in r["text"].lower() for a in g["answer"]),
                         "citations_ok": bool(v.get("ok")) and not abstained,
                         "cited_right_doc": g["doc"] in cited_titles, "answer": r["text"][:300]})
            if r["stats"].get("tokens_per_second"):
                speeds.append(r["stats"]["tokens_per_second"])
                firsts.append(r["stats"].get("first_token_seconds", 0))
        ans = [x for x in rows if x["answerable"]]
        un = [x for x in rows if not x["answerable"]]
        report["answers"] = {
            "correct": sum(x["correct"] for x in ans) / len(ans),
            "citations_ok": sum(x["citations_ok"] for x in ans) / len(ans),
            "cited_right_doc": sum(x["cited_right_doc"] for x in ans) / len(ans),
            "false_abstention": sum(x["abstained"] for x in ans) / len(ans),
            "abstained_when_unanswerable": sum(x["abstained"] for x in un) / max(len(un), 1)}
        report["speed"] = {"tokens_per_second": round(statistics.mean(speeds), 1) if speeds else None,
                           "first_token_seconds": round(statistics.mean(firsts), 2) if firsts else None}
        tool_ok = 0
        for prompt, expected in TOOL_CASES:
            r = await ask(engine, message=prompt)
            called = any(t["name"] == "calculator" for t in r["tools"])
            tool_ok += called and expected in r["text"].replace(",", "")
        report["tools"] = {"calculator_called_and_correct": tool_ok / len(TOOL_CASES)}
        report["rows"] = rows

    # ---- print ------------------------------------------------------------
    i = report["indexing"]
    print(f"\nadi-gpt evaluation ({report['mode']}; embeddings: "
          f"{'fake hash' if args.offline else cfg.embed_model})")
    print(f"Indexed {i['files']} files into {i['chunks']} chunks in {i['seconds']}s; "
          f"re-index of unchanged files {i['reindex_seconds']}s ({i['reindex_unchanged']} skipped)")
    print(f"\nRetrieval over {len(answerable)} answerable questions")
    print(f"{'method':8} {'hit@1':>6} {'hit@3':>6} {'hit@6':>6} {'MRR':>6}")
    for method, m in report["retrieval"].items():
        print(f"{method:8} {pct(m['hit@1']):>6} {pct(m['hit@3']):>6} {pct(m['hit@6']):>6} {m['mrr']:>6.2f}")
    g = report["gate"]
    print(f"\nAbstention gate: blocked {pct(g['unanswerable_blocked'])} of {len(unanswerable)} "
          f"unanswerable questions before the model; wrongly blocked "
          f"{pct(g['answerable_wrongly_blocked'])} of answerable ones")
    if "answers" in report:
        a = report["answers"]
        print(f"\nAnswers with {cfg.chat_model} (strict, documents on)")
        print(f"  correct: {pct(a['correct'])}   citations pass check: {pct(a['citations_ok'])}   "
              f"cited the right file: {pct(a['cited_right_doc'])}")
        print(f"  abstained when unanswerable: {pct(a['abstained_when_unanswerable'])}   "
              f"false abstention: {pct(a['false_abstention'])}")
        print(f"  calculator tool called and correct: "
              f"{pct(report['tools']['calculator_called_and_correct'])}")
        print(f"  speed: {report['speed']['tokens_per_second']} tok/s, "
              f"first token {report['speed']['first_token_seconds']}s")
    out = ROOT / f"results-{report['mode']}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")
    await ollama.close()
    db.close()
    tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
