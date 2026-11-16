"""realtime_answer.py — answer one question with LFM3-Omni over /v1/realtime, text only, with optional web search.

answer(question, on_text, on_status, on_sources, on_announce, history=()) -> full answer text

- Factual questions (needs_search) take the search path:
    1. the announcement is spoken ("A pesquisar na internet") before any voice session opens;
    2. a session with the web_search tool asks the model for a query;
    3. that session is closed, the search runs, and the results go to on_sources;
    4. a second session, with no tools, answers from the results as plain text.
  Two sessions because the server runs one at a time and refuses to change tools mid-session.
- Other questions are answered in one session without tools.

The caller speaks the text with its own voices. The language of the answer follows the question.
"""
import json
import re
from collections.abc import Awaitable, Callable, Sequence

import websockets

from voice_chat import resolve_realtime_url
from web_search import TOOL_SPEC, search

# instructions per language of the question; the answer follows the question
RULES = {
    "pt": {
        "lang": "Responde sempre em português europeu (pt-PT), não em português do Brasil nem em inglês. ",
        "announce": "A pesquisar na internet, um momento.",
        "after_search": "Responde à pergunta usando só os resultados da pesquisa, em duas frases.",
    },
    "en": {
        "lang": "Always answer in English. ",
        "announce": "Searching the web, one moment.",
        "after_search": "Answer the question using only the search results, in two sentences.",
    },
}

# words that mean the answer may have changed since the model was trained
SEARCH_HINTS = re.compile(
    r"\d|quant[oa]s?\b|quando|hoje|agora|atual|actual|último|ultim|notícia|noticia|"
    r"preço|preco|cotação|cotacao|tempo\b|previs|resultado|ganhou|jogo|ao vivo|"
    r"\bhow many\b|\bwhen\b|today|latest|current|price|weather|score|news|who won|population|released|release",
    re.IGNORECASE,
)


def needs_search(question: str) -> bool:
    return bool(SEARCH_HINTS.search(question))


def _with_history(question: str, history: Sequence[tuple[str, str]]) -> str:
    """The question, preceded by the earlier turns of this conversation, if any."""
    if not history:
        return question
    lines = [f"Pergunta: {q}\nResposta: {a}" for q, a in history]
    return "Conversa até agora:\n" + "\n\n".join(lines) + f"\n\nNova pergunta: {question}"


async def _send(ws, obj):
    await ws.send(json.dumps(obj))


async def _user_text(ws, text):
    await _send(ws, {"type": "conversation.item.create", "item": {
        "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})


async def _collect(ws, on_text: Callable[[str], None]) -> tuple[str, list[dict]]:
    """Read events until response.done; returns (text, function-call events)."""
    text, calls = "", []
    while True:
        ev = json.loads(await ws.recv())
        t = ev["type"]
        if t == "response.output_text.delta":
            text += ev["delta"]
            on_text(ev["delta"])
        elif t == "response.function_call_arguments.done":
            calls.append(ev)
        elif t == "error":
            raise RuntimeError(ev["error"]["message"])
        elif t == "response.done":
            return text, calls


async def _open(url: str):
    ws = await websockets.connect(url, max_size=None, open_timeout=10)
    await ws.recv()  # session.created
    return ws


async def answer(question: str,
                 on_text: Callable[[str], None],
                 on_status: Callable[[str], None],
                 on_sources: Callable[[list], None],
                 on_announce: Callable[[str], Awaitable[None]],
                 url: str | None = None,
                 history: Sequence[tuple[str, str]] = (),
                 lang: str | None = None,
                 on_phase: Callable[[str], None] | None = None) -> str:
    # the language Whisper heard the question in, when it said; the question's own text otherwise
    rules = RULES.get(_rules_lang(lang) or _lang(question), RULES["pt"])
    url = url or resolve_realtime_url()

    if not needs_search(question):
        ws = await _open(url)
        try:
            await _send(ws, {"type": "session.update", "session": {
                "type": "realtime", "output_modalities": ["text"]}})
            await ws.recv()
            await _user_text(ws, rules["lang"] + _with_history(question, history))
            await _send(ws, {"type": "response.create"})
            text, _ = await _collect(ws, on_text)
            return text
        finally:
            await ws.close()

    # 1. the announcement, before any voice session is open
    on_status("A pesquisar na internet…")
    await on_announce(rules["announce"])

    # 2. the model asks for a query, with the search tool
    ws = await _open(url)
    try:
        await _send(ws, {"type": "session.update", "session": {
            "type": "realtime", "output_modalities": ["text"],
            "tools": [TOOL_SPEC], "tool_choice": "required"}})
        await ws.recv()
        await _user_text(ws, _with_history(question, history))
        await _send(ws, {"type": "response.create"})
        _, calls = await _collect(ws, lambda _: None)
    finally:
        await ws.close()
    if not calls:
        raise RuntimeError("model did not call web_search")
    query = json.loads(calls[0].get("arguments") or "{}").get("query", question)

    if on_phase: on_phase("search")          # the octopus shows a magnifier while the web is searched
    # 3. the search itself
    results = search(query, n=5)
    on_sources(results)
    on_status("A responder…")
    if on_phase: on_phase("write")           # the octopus types while the answer is written

    # 4. the answer from the results, in a session with no tools
    ws = await _open(url)
    try:
        await _send(ws, {"type": "session.update", "session": {
            "type": "realtime", "output_modalities": ["text"]}})
        await ws.recv()
        results_text = "\n".join(f"- {r['title']}: {r['snippet']}" for r in results)
        await _user_text(ws, rules["lang"] + rules["after_search"]
                         + f"\n\nPergunta: {question}\nResultados da pesquisa para «{query}»:\n{results_text}")
        await _send(ws, {"type": "response.create"})
        text, _ = await _collect(ws, on_text)
        return text
    finally:
        await ws.close()


def _rules_lang(whisper_language: str | None) -> str | None:
    """Whisper's language name ('portuguese', 'english') as a rules key ('pt', 'en'), or None."""
    if not whisper_language:
        return None
    name = whisper_language.lower()
    if name.startswith("port"):
        return "pt"
    if name.startswith("eng"):
        return "en"
    return None


def _lang(question: str) -> str:
    from speech import lang_of
    return lang_of(question)
