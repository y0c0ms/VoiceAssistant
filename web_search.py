"""web_search.py — web search for the voice assistant, through the SearXNG instance the herdr voice
agent uses (herdr-desk-deck/tools/herdr_voice/search.py: same URL, same JSON parameters).

search(query, n) -> list of {"title", "link", "snippet"}
"""
import json
import os
import urllib.parse
import urllib.request

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:30850")
TIMEOUT_S = 15.0
UA = "voiceassistant/0.1"


def search_url(query: str) -> str:
    # same parameters as herdr_voice.search.search_url: JSON format, auto language, no safesearch filter
    return (f"{SEARXNG_URL}/search?format=json&language=auto&safesearch=0"
            f"&q={urllib.parse.quote(query)}")


def search(query: str, n: int = 5, timeout: float = TIMEOUT_S) -> list[dict]:
    req = urllib.request.Request(search_url(query), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8", "replace"))
    out = []
    for item in payload.get("results", []):
        title = (item.get("title") or "").strip()
        snippet = (item.get("content") or "").strip()
        link = item.get("url") or ""
        # keep results in Latin script (pt, en, es, fr...); foreign-script pages answer other questions
        if not link or _foreign_share(title + " " + snippet) > 0.15:
            continue
        out.append({"title": title, "link": link, "snippet": snippet})
        if len(out) >= n:
            break
    return out


def _foreign_share(text: str) -> float:
    """Share of letters that are not Latin: CJK, Cyrillic, Arabic, Hangul, Thai, Devanagari."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    foreign = sum(1 for c in letters if ord(c) > 0x024F)
    return foreign / len(letters)


TOOL_SPEC = {
    "type": "function",
    "name": "web_search",
    "description": "Search the web for current or factual information. Use it for news, "
                   "prices, weather, schedules, or anything that may have changed.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "search query"}},
        "required": ["query"],
    },
}
