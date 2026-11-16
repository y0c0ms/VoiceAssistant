"""herdr_pane.py: the herdr panes as a tool for the assistant. Built on the herdr CLI (`herdr pane ...`).

- list_panes(), read_pane(): read-only; the assistant may use them at any time.
- stage_send(): a prompt for a pane is only recorded, never sent. The card reads it back ("Enviar para <pane>: ...?
  Diz sim ou não.") and confirm() sends it, or cancel() drops it. Sending is a change in another program, so it always
  waits for a yes, as the herdr voice agent does for anything with a consequence.
"""
import json
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
