"""Optional web search: DuckDuckGo-style search via `ddgs`, pages cleaned with trafilatura.

Queries leave the machine when this is used, so it is off unless toggled per message.
"""
from __future__ import annotations

import asyncio

import httpx

PAGE_CHARS = 2500
UA = "Mozilla/5.0 (X11; Linux x86_64) adi-gpt/0.1 (personal assistant)"


def _search_sync(query: str, n: int) -> list[dict]:
    from ddgs import DDGS
    results = DDGS().text(query, max_results=n)
    return [{"title": r.get("title") or r.get("href", ""), "url": r.get("href", ""),
             "text": r.get("body", "")} for r in results if r.get("href")]


async def _fetch(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            return ""
        import trafilatura
        text = await asyncio.to_thread(trafilatura.extract, r.text)
        return (text or "")[:PAGE_CHARS]
    except Exception:  # noqa: BLE001 - a failed page just falls back to its snippet
        return ""


async def web_search(query: str, n: int = 4, fetch_pages: bool = True) -> list[dict]:
    """Return sources shaped like document hits: title, url, text, kind='web'."""
    results = await asyncio.to_thread(_search_sync, query, n)
    if fetch_pages and results:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True,
                                     headers={"User-Agent": UA}) as client:
            pages = await asyncio.gather(*[_fetch(client, r["url"]) for r in results])
        for r, page in zip(results, pages):
            if len(page) > len(r["text"]):
                r["text"] = page
    return [{**r, "heading": "", "kind": "web"} for r in results if r["text"]]
