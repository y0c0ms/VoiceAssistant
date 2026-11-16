"""specialist.py: the specialist the card calls for hard tasks, and the router that decides who answers.

Backends (VA_SPECIALIST):
- opencode (default): the opencode CLI with a free model (opencode/deepseek-v4.1-flash-free), run in
  SPECIALIST_DIR with the `build` agent (full tools; files it writes land in that folder). One opencode
  session per card, so the conversation is kept in that folder and follow-ups resume it. Nothing is loaded
  on the GPU, so nothing is unloaded.
- qwen-local: Qwen through llama-swap. Loading it unloads the voice models (exclusive `gpu` group), so the
  card says it is loading BEFORE the load starts and that it is ready AFTER it.

Routing (decide): the keywords in SPECIALIST_RE go to the specialist; a question the keywords do not match
is classified by the front desk (LFM3); a card already with the specialist keeps its follow-ups there until
the user asks for the assistant back (SWITCH_BACK_RE).
"""
import asyncio
import difflib
import json
import os
import re
import subprocess
import urllib.request
import config
from collections.abc import Awaitable, Callable

_cfg = config.load()
_spec = _cfg["providers"]["specialist"]
BACKEND = _spec["backend"]
SWAP = _cfg["providers"]["front_desk"]["url"]
FRONT_MODEL = _cfg["providers"]["front_desk"]["model"]     # the quick front desk; also the classifier
LOCAL_MODEL = _spec["local_model"]
OPENCODE = "opencode"
OPENCODE_MODEL = _spec["opencode_model"]
OPENCODE_AGENT = _spec["opencode_agent"]                   # full tools; writes land in SPECIALIST_DIR
SPECIALIST_DIR = os.path.expanduser(_spec["opencode_dir"])
NAME = "DeepSeek" if BACKEND == "opencode" else "Qwen"
LOAD_TIMEOUT_S = 330      # llama-swap healthCheckTimeout is 300 s; the load call returns when the model answers
ASK_TIMEOUT_S = 300
MAX_ANSWER_TOKENS = 900

SPECIALIST_RE = re.compile(r"\b(qwen|especialista|specialist|diagram\w*|diagrama\w*|mermaid)\b", re.IGNORECASE)
SWITCH_BACK_RE = re.compile(
    r"\b(volta ao assistente|voltar ao assistente|fala com o assistente|back to the assistant|switch back|volta ao lfm)\b",
    re.IGNORECASE)

# opencode's own system prompt is a coding agent's; this line keeps replies short and speakable
GUIDE = ("Answer the user's question directly, in plain spoken language, in English or European Portuguese only. "
         "Keep it short. Do not propose a plan and do not ask for approval. If asked to make or change a file, "
         "do it in the current folder and name the file. A diagram is never written as text: write one HTML file in "
         "the current folder, named diagram-<short-name>.html, that shows the diagram with mermaid from "
         "https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js, and reply with that file name only.")
SYSTEM_PROMPT = ("You are the specialist behind a voice assistant. Your answer is read aloud: write plain "
                 "sentences, no markdown tables, no emojis. Answer in English or European Portuguese only, concisely.")
# LFM3 ignores a system role here and answers the question, so the instruction and the question share one user message
CLASSIFY_TEMPLATE = ("Classify the question below. Answer with exactly one word: QWEN or LFM3.\n"
                     "QWEN = diagrams, flowcharts, schemas, code, planning, detailed technical explanations, comparisons, "
                     "and anything that needs files, folders, the browser, apps or the computer (find, open, read, show).\n"
                     "LFM3 = quick facts, time, weather, translation, small talk, recent news.\n"
                     "Examples: 'Draw a flowchart of a login' -> QWEN. 'Write a script that parses CSV' -> QWEN. "
                     "'Find and open the Outbox folder' -> QWEN. 'Open the diagram in my browser' -> QWEN. "
                     "'Explain in detail how a router works' -> QWEN. 'What time is it in Lisbon?' -> LFM3. "
                     "'Tell me a joke' -> LFM3.\n"
                     "Question: {q}\nAnswer:")

LINES = {
    "pt": {"loading": "A carregar o modelo Qwen. Pode demorar um pouco.", "loaded": "Modelo Qwen carregado."},
    "en": {"loading": "Loading the Qwen model. This can take a moment.", "loaded": "Qwen model loaded."},
}


def needs_specialist(question: str) -> bool:
    return bool(SPECIALIST_RE.search(question))


def wants_assistant(question: str) -> bool:
    """True when the user asks to go back to the front desk (LFM3)."""
    return bool(SWITCH_BACK_RE.search(question))


def parse_verdict(reply: str) -> bool:
    return reply.strip().upper().startswith("QWEN")


SYSTEM_ONE_URL = _cfg["providers"]["router"]["url"]
ROUTE_CRITERIA = {
    "front": "quick facts, the time, weather, translation, small talk, recent news, what is on the screen",
    "specialist": "diagrams, flowcharts, code, planning, detailed explanations, comparisons, and anything that needs "
                  "files, folders, the browser, apps or the computer",
}


def _route_systemone(question: str) -> bool | None:
    """The System One router (CPU, one forward pass, about 0.1 s): True for the specialist. None if it is down."""
    body = json.dumps({"state": question, "questions": {"route": {
        "type": "choice", "instructions": "Which agent should answer this voice request?",
        "criteria": ROUTE_CRITERIA}}}).encode()
    req = urllib.request.Request(SYSTEM_ONE_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            choice = json.loads(r.read().decode())["answers"]["route"].get("choice")
    except Exception as e:
        print(f"system-one router unavailable: {e}", flush=True)
        return None
    return None if choice is None else choice == "specialist"


async def classify(question: str) -> bool:
    """Does the specialist answer this? System One decides; if it is down, LFM3 decides. A failure of both keeps
    the question with the front desk."""
    verdict = await asyncio.to_thread(_route_systemone, question)
    if verdict is not None:
        return verdict
    body = {"model": FRONT_MODEL,
            "messages": [{"role": "user", "content": CLASSIFY_TEMPLATE.format(q=question)}],
            "max_tokens": 4, "temperature": 0}
    try:
        reply = await asyncio.to_thread(_post, body, 30)
    except Exception as e:
        print(f"classify failed, the front desk answers: {e}", flush=True)
        return False
    return parse_verdict(reply)


async def decide(question: str, sticky: bool) -> bool:
    """True when the specialist answers this question. `sticky`: the previous turn went to the specialist."""
    if wants_assistant(question):
        return False
    if needs_specialist(question) or sticky:
        return True
    return await classify(question)


def _post(body: dict, timeout: float) -> str:
    req = urllib.request.Request(f"{SWAP}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"].get("content") or ""


def model_loaded() -> bool:
    """qwen-local only: True when llama-swap reports the local specialist model as loaded."""
    with urllib.request.urlopen(f"{SWAP}/v1/models", timeout=10) as r:
        models = json.loads(r.read().decode())["data"]
    return any(m["id"] == LOCAL_MODEL and m.get("status", {}).get("value") == "loaded" for m in models)


def load_model() -> None:
    """qwen-local only: the smallest request; llama-swap starts the model and answers once it is ready."""
    _post({"model": LOCAL_MODEL, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1},
          LOAD_TIMEOUT_S)




SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv"}
FOLDER_TEMPLATE = ("The user asks an assistant to work with a folder. Reply with only the folder's name as the user "
                   "said it, or NONE if no folder is named.\n"
                   "Examples: 'Open the Outbox folder' -> Outbox. 'Read my projects folder' -> projects. "
                   "'What time is it?' -> NONE.\n"
                   "Request: {q}\nName:")


def write_permissions(folders: list[str]) -> None:
    """The specialist reads the approved folders and its own, and writes only in its own folder: approved folders are
    read-only, and no shell runs. Written before every request, so a folder approved by voice counts at once.
    opencode takes the first rule that matches, so the allows come before the catch-all, and the approved folders'
    write denies come before the catch-all allow."""
    approved = [os.path.realpath(os.path.expanduser(f)) for f in folders]
    own = os.path.realpath(SPECIALIST_DIR)
    reach = {p: "allow" for f in [*approved, own] for p in (f, f"{f}/**")}
    reach["*"] = "deny"
    writes = {p: "deny" for f in approved for p in (f, f"{f}/**")}
    writes["*"] = "allow"
    with open(os.path.join(SPECIALIST_DIR, "opencode.json"), "w") as f:
        json.dump({"$schema": "https://opencode.ai/config.json",
                   "permission": {"external_directory": reach, "edit": writes, "bash": {"*": "deny"}}},
                  f, indent=2)


def find_folders(name: str, root: str = os.path.expanduser("~"), max_depth: int = 5) -> list[str]:
    """Folders under `root` whose name is the one the user said (accents, case and punctuation ignored)."""
    want = config.normalize(name)
    hits = []
    base = root.rstrip("/").count("/")
    for dirpath, dirnames, _ in os.walk(root):
        depth = dirpath.count("/") - base
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS] if depth < max_depth else []
        for d in dirnames:
            got = config.normalize(d)
            if got == want or (len(want) >= 4 and difflib.SequenceMatcher(None, got, want).ratio() >= 0.75):
                hits.append(os.path.join(dirpath, d))
    return hits


async def folder_named(question: str) -> str | None:
    """The folder the request names, as LFM3 reads it, or None."""
    body = {"model": FRONT_MODEL,
            "messages": [{"role": "user", "content": FOLDER_TEMPLATE.format(q=question)}],
            "max_tokens": 12, "temperature": 0}
    try:
        reply = await asyncio.to_thread(_post, body, 30)
    except Exception as e:
        print(f"folder_named failed: {e}", flush=True)
        return None
    name = reply.strip().strip("\"'.").strip()
    return None if not name or name.upper() == "NONE" else name


class Session:
    """One card's conversation with the specialist: the opencode session id (or the local history)."""

    def __init__(self) -> None:
        self.sid: str | None = None
        self.history: list[tuple[str, str]] = []

    def ask(self, question: str) -> str:
        return self._ask_opencode(question) if BACKEND == "opencode" else self._ask_local(question)

    def _ask_opencode(self, question: str) -> str:
        os.makedirs(SPECIALIST_DIR, exist_ok=True)
        write_permissions(config.load()["access"]["folders"])
        cmd = [OPENCODE, "run", "-m", OPENCODE_MODEL, "--agent", OPENCODE_AGENT,
               "--dir", SPECIALIST_DIR, "--format", "json"]
        if self.sid:
            cmd += ["--session", self.sid]
        cmd.append(f"{GUIDE}\n\n{question}")
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=ASK_TIMEOUT_S)
        texts: list[str] = []
        for line in p.stdout.splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.sid = self.sid or e.get("sessionID")
            if e.get("type") == "text":
                texts.append(e["part"]["text"])
        if p.returncode != 0 or not texts:
            raise RuntimeError((p.stderr or p.stdout).strip()[-200:] or f"exit {p.returncode}")
        return "\n".join(texts).strip()

    def _ask_local(self, question: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for q, a in self.history:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": question})
        text = _post({"model": LOCAL_MODEL, "messages": messages, "max_tokens": MAX_ANSWER_TOKENS},
                     ASK_TIMEOUT_S).strip()
        self.history.append((question, text))
        return text


async def answer(question: str, session: Session, on_status: Callable[[str], None],
                 announce: Callable[[str], Awaitable[None]]) -> str:
    """Answer a question with the specialist. The local model, if used, is loaded first and the card
    says so out loud before and after; the opencode backend has nothing to load."""
    from speech import lang_of
    lines = LINES["pt" if lang_of(question) == "pt" else "en"]
    try:
        if BACKEND == "qwen-local" and not await asyncio.to_thread(model_loaded):
            on_status(lines["loading"])
            await announce(lines["loading"])          # said first; the load starts only after this
            await asyncio.to_thread(load_model)
            on_status(lines["loaded"])
            await announce(lines["loaded"])
        on_status(f"{NAME} a responder…")
        return await asyncio.to_thread(session.ask, question)
    except Exception as e:                            # a failed call is shown, not raised
        message = f"Não consegui usar o {NAME}: {e}"
        on_status(message)
        return message
