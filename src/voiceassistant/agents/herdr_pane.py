"""herdr_pane.py: the herdr terminal panes as tools for the front desk. Built on the herdr CLI (`herdr pane ...`).

- list_panes and read_pane are read-only: the model may call them whenever a question is about the panes.
- send_to_pane only stages a prompt (stage_send). Nothing is sent until the user says yes: the card reads it back
  ("Enviar para <pane>: ...? Diz sim ou não."), then confirm() sends it or cancel() drops it. Sending changes another
  program, so it always waits for a yes.
"""
import json
import re
import subprocess
import unicodedata

HERDR = "herdr"
_pending: dict | None = None


class PaneError(RuntimeError):
    pass


def _run(*args: str, timeout: float = 10) -> str:
    p = subprocess.run([HERDR, *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise PaneError((p.stderr or p.stdout).strip() or f"herdr exited {p.returncode}")
    return p.stdout


def list_panes() -> list[dict]:
    data = json.loads(_run("pane", "list"))
    if "error" in data:
        raise PaneError(data["error"].get("message", "herdr error"))
    return [{"id": p["pane_id"],
             "title": p.get("terminal_title_stripped") or p.get("cwd", ""),
             "cwd": p.get("cwd", ""),
             "focused": bool(p.get("focused")),
             "agent": p.get("agent_status", "unknown")}
            for p in data["result"]["panes"]]


def read_pane(pane_id: str, lines: int = 60) -> str:
    return _run("pane", "read", pane_id, "--source", "recent", "--lines", str(lines), "--format", "text")


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def find_panes(query: str) -> list[dict]:
    """Panes whose title or folder contains the words said (accents and case ignored)."""
    words = _plain(query).split()
    return [p for p in list_panes()
            if words and all(w in _plain(p["title"] + " " + p["cwd"]) for w in words)]


def stage_send(pane_id: str, text: str) -> dict:
    """Record the prompt for a pane; nothing is sent until confirm()."""
    global _pending
    pane = next((p for p in list_panes() if p["id"] == pane_id), None)
    if pane is None:
        raise PaneError(f"no pane {pane_id}")
    _pending = {"id": pane_id, "title": pane["title"], "text": text}
    return dict(_pending)


def pending() -> dict | None:
    return dict(_pending) if _pending else None


def readback(p: dict) -> str:
    return f"Enviar para {p['title']}: {p['text']}? Diz sim ou não."


def confirm() -> str:
    """Send the staged prompt (text, then Enter). Returns the pane it went to."""
    global _pending
    if _pending is None:
        raise PaneError("nothing is waiting for a yes")
    p, _pending = _pending, None
    _run("pane", "run", p["id"], p["text"])
    return p["id"]


def cancel() -> None:
    global _pending
    _pending = None


# What a pane request says, in Portuguese. A small model picks tools unreliably, so the request is read here and the
# action runs here; the model only phrases the answer from what the action returned.
SEND_RE = re.compile(
    r"\b(?:envia|enviar|manda|mandar|diz|dizer|escreve|escrever)\s+(?:ao|à|a|para\s+(?:o|a))\s+"
    r"(?:pain[eé]l\s+)?(?P<pane>[^,:?]+?)\s*(?:[,:]|\s+que\s+)\s*(?P<text>.+?)\s*$", re.IGNORECASE)
READ_RE = re.compile(
    r"\b(?:lê|le|lê-me|mostra|mostrar|resume|resumir|o\s+que\s+(?:está|esta|tem))\b.*?\bpain(?:el|éis|eis)\s+"
    r"(?P<pane>[^?.!]+?)\s*[?.!]*\s*$", re.IGNORECASE)
LIST_RE = re.compile(r"\b(?:lista|listar|quais|mostra|mostrar)\b.*\bpain(?:el|éis|eis)\b|\bpain[eé]is\s+abertos\b",
                     re.IGNORECASE)


def parse_request(question: str) -> tuple[str, dict] | None:
    """What the user asked of the panes: ("send", {pane, text}), ("read", {pane}), ("list", {}), or None."""
    q = question.strip()
    m = SEND_RE.search(q)
    if m:
        return "send", {"pane": m["pane"].strip(), "text": m["text"].strip()}
    m = READ_RE.search(q)
    if m:
        return "read", {"pane": m["pane"].strip()}
    if LIST_RE.search(q):
        return "list", {}
    return None


def _one_pane(query: str) -> tuple[dict | None, str]:
    panes = find_panes(query)
    if not panes:
        return None, f"Não encontrei nenhum painel para «{query}»."
    if len(panes) > 1:
        return None, "Há mais do que um painel com esse nome: " + ", ".join(p["title"] for p in panes) + "."
    return panes[0], ""


def run_tool(name: str, args: dict) -> str:
    """Run one pane action (list_panes, read_pane, send_to_pane) and return its text. It calls herdr: run it in a thread."""
    try:
        if name == "list_panes":
            return json.dumps([{"title": p["title"], "folder": p["cwd"]} for p in list_panes()], ensure_ascii=False)
        pane, err = _one_pane(str(args.get("pane", "")))
        if pane is None:
            return err
        if name == "read_pane":
            return read_pane(pane["id"])[-2000:]
        if name == "send_to_pane":
            stage_send(pane["id"], str(args.get("text", "")))
            return "Preparado. Só é enviado depois de o utilizador dizer sim; não digas que já foi enviado."
        return f"Ferramenta desconhecida: {name}."
    except PaneError as e:
        return f"O herdr não respondeu: {e}"
