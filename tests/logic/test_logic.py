"""Logic tests for the voice overlay. No audio, no screen, no network: they run in a second.

    python3 tests/logic/test_logic.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

import numpy as np  # noqa: E402

from voiceassistant import app as vo  # noqa: E402
from voiceassistant.core.stop import is_stop_request  # noqa: E402
from voiceassistant.ui.rich_text import plain  # noqa: E402
from voiceassistant.answers.realtime_answer import _with_history, needs_search  # noqa: E402
from voiceassistant.answers.screen import needs_screen, pick_side  # noqa: E402
from voiceassistant.answers import web_search  # noqa: E402
from voiceassistant.agents import specialist  # noqa: E402


# --- Qwen specialist routing ----------------------------------------------------------------------

def test_specialist_routing():
    check("specialist: qwen named", specialist.needs_specialist("Ask Qwen about this"), "")
    check("specialist: especialista", specialist.needs_specialist("Pergunta ao especialista"), "")
    check("specialist: diagram", specialist.needs_specialist("Draw a diagram of the flow"), "")
    check("specialist: diagrama", specialist.needs_specialist("Faz um diagrama da rede"), "")
    check("specialist: mermaid", specialist.needs_specialist("Show me in mermaid"), "")
    check("not specialist: plain question", not specialist.needs_specialist("What is the capital of Portugal?"), "")
    check("not specialist: search question", not specialist.needs_specialist("When was Unreal 5.9 released?"), "")
    check("not specialist: screen question", not specialist.needs_specialist("What is on my left screen?"), "")


def test_router_decide():
    """Keywords and switch-back are decided without asking the classifier; other questions go to LFM3's verdict."""
    import asyncio
    calls: list[str] = []
    async def fake_classify(q):
        calls.append(q)
        return q.startswith("Write")
    real_classify = specialist.classify
    specialist.classify = fake_classify
    try:
        run = lambda q, sticky: asyncio.run(specialist.decide(q, sticky))
        check("router: keyword goes to specialist", run("Draw a diagram of the flow", False), "")
        check("router: sticky follow-up stays", run("And what about Porto?", True), "")
        check("router: switch back to assistant", not run("Volta ao assistente", True), "")
        check("router: classifier says QWEN for a flowchart request",
              run("Write a flowchart of the login process", False), "")
        check("router: classifier says LFM3 for a quick fact",
              not run("What is the capital of Portugal?", False), "")
        check("router: classifier not asked when keyword or sticky decides",
              calls == ["Write a flowchart of the login process", "What is the capital of Portugal?"], str(calls))
    finally:
        specialist.classify = real_classify


def test_stop_inside_echo():
    """A stop word spoken over the answer must survive the echo check; pure echo must leave nothing."""
    spoken = "The capital of Portugal is Lisbon."
    check("echo with a stop word leaves the stop", vo.residual_words("The capital of Portugal stop", spoken).strip() == "stop",
          vo.residual_words("The capital of Portugal stop", spoken))
    check("pure echo leaves nothing", vo.residual_words("The capital of Portugal is Lisbon", spoken) == "", "")
    check("stop inside echo is a stop request", is_stop_request(vo.residual_words("Lisbon stop", spoken)), "")


def test_verdict_parsing():
    check("verdict: QWEN", specialist.parse_verdict("QWEN"), "")
    check("verdict: lowercase qwen with spaces", specialist.parse_verdict("  qwen\n"), "")
    check("verdict: LFM3", not specialist.parse_verdict("LFM3"), "")
    check("verdict: unexpected text is the front desk", not specialist.parse_verdict("I think"), "")


def test_session_resumes_opencode_session():
    """The second question must pass the session id from the first answer, so the conversation resumes."""
    import json as _json
    real_run = specialist.subprocess.run
    real_backend = specialist.BACKEND
    seen: list[list[str]] = []
    def fake_run(cmd, **kw):
        seen.append(cmd)
        sid = "ses_test" if len(seen) == 1 else None
        lines = [{"type": "step_start", "sessionID": "ses_test"},
                 {"type": "text", "sessionID": "ses_test", "part": {"text": f"answer {len(seen)}"}}]
        class P: returncode = 0; stdout = "\n".join(_json.dumps(l) for l in lines); stderr = ""
        return P()
    specialist.subprocess.run = fake_run
    specialist.BACKEND = "opencode"
    try:
        s = specialist.Session()
        first = s.ask("Draw a diagram")
        second = s.ask("And the next step?")
        check("session: first answer text", first == "answer 1", first)
        check("session: follow-up gets the answer text", second == "answer 2", second)
        check("session: first call starts a new session", "--session" not in seen[0], str(seen[0]))
        check("session: follow-up resumes the same session",
              seen[1][seen[1].index("--session") + 1] == "ses_test", str(seen[1]))
        check("session: runs the build agent in the specialist folder",
              seen[0][seen[0].index("--agent") + 1] == "build"
              and seen[0][seen[0].index("--dir") + 1] == specialist.SPECIALIST_DIR, str(seen[0]))
    finally:
        specialist.subprocess.run = real_run
        specialist.BACKEND = real_backend


def test_specialist_announcement_order():
    """qwen-local: the loading line must be spoken BEFORE the model load starts."""
    import asyncio
    events: list[str] = []
    async def fake_announce(t): events.append("say:" + t)
    async def fake_status(t): events.append("status:" + t)
    real_backend, real_model_loaded, real_load = specialist.BACKEND, specialist.model_loaded, specialist.load_model
    real_ask = specialist.Session.ask
    try:
        specialist.BACKEND = "qwen-local"
        specialist.model_loaded = lambda: False
        def fake_load(): events.append("load:start")
        specialist.load_model = fake_load
        specialist.Session.ask = lambda self, q: "the answer"
        result = asyncio.run(specialist.answer("Draw a diagram", specialist.Session(), fake_status, fake_announce))
        check("specialist: loading said before load starts",
              events.index("say:" + specialist.LINES["en"]["loading"]) < events.index("load:start"), str(events))
        check("specialist: loaded said after load",
              events.index("say:" + specialist.LINES["en"]["loaded"]) > events.index("load:start"), str(events))
        check("specialist: answer returned", result == "the answer", result)
    finally:
        specialist.BACKEND, specialist.model_loaded, specialist.load_model = real_backend, real_model_loaded, real_load
        specialist.Session.ask = real_ask


# --- octopus: colour switch, hat, text box -----------------------------------------------------------

def test_switch_timeline_order():
    """To the specialist: colour first, hat after. Back to the assistant: hat off first, then colour."""
    from voiceassistant.ui import mascot as po2
    c, h = po2.COLOUR_S, po2.HAT_S
    st = po2.timeline(0.0, True)
    check("to specialist: colour starts at 0", st["progress"] == 0.0 and st["hat"] == 0.0, str(st))
    st = po2.timeline(c * 0.5, True)
    check("to specialist: colour half way, no hat", 0.4 < st["progress"] < 0.6 and st["hat"] == 0.0, str(st))
    st = po2.timeline(c + h * 0.5, True)
    check("to specialist: colour done, hat half landed", st["progress"] == 1.0 and 0.4 < st["hat"] < 0.6, str(st))
    st = po2.timeline(0.0, False)
    check("to assistant: hat still on at start", st["hat"] == 1.0 and st["progress"] == 0.0, str(st))
    st = po2.timeline(h * 0.5, False)
    check("to assistant: hat coming off, colour not moved", 0.4 < st["hat"] < 0.6 and st["progress"] == 0.0, str(st))
    st = po2.timeline(h + c * 0.5, False)
    check("to assistant: hat off, colour half way", st["hat"] == 0.0 and 0.4 < st["progress"] < 0.6, str(st))
    ages = [i * 0.05 for i in range(60)]
    prog = [po2.timeline(a, True)["progress"] for a in ages]
    check("colour wave never goes back", all(b >= a for a, b in zip(prog, prog[1:])))


def test_random_seed_point():
    import math as _m
    import random as _r
    from voiceassistant.ui import mascot as po2
    seeds = [po2.random_seed_point(_r.Random(s)) for s in range(40)]
    check("seed is on the head", all((x - 80) ** 2 + (y - 40) ** 2 <= 20 ** 2 + 1e-6 for x, y in seeds), "")
    check("different rng seeds give different start points", len(set(seeds)) > 30, str(len(set(seeds))))
    check("same rng seed gives the same start point", po2.random_seed_point(_r.Random(5))
          == po2.random_seed_point(_r.Random(5)), "")


def test_colour_wave_covers_body():
    """The wave starts at the seed and reaches more of the octopus as progress rises, all of it at 1."""
    import random as _r
    import numpy as np
    from voiceassistant.ui import mascot as po2
    seed = po2.random_seed_point(_r.Random(11))
    a_only = po2.render_frame("assistant", "assistant", 1.0, None, 0.0, "idle", 0.3, 40.0)
    b_only = po2.render_frame("specialist", "specialist", 1.0, None, 0.0, "idle", 0.3, 40.0)
    shape = (a_only[..., 3] > 0)
    fracs = []
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        frame = po2.render_frame("assistant", "specialist", p, seed, 0.0, "idle", 0.3, 40.0)
        differs = shape & np.any(a_only[..., :3] != b_only[..., :3], axis=2)
        fracs.append(float(np.mean(np.all(frame[differs, :3] == b_only[differs, :3], axis=1))))
    check("wave: nothing changed at progress 0", fracs[0] == 0.0, str(fracs))
    check("wave: spreads as progress rises", all(b >= a for a, b in zip(fracs, fracs[1:])), str(fracs))
    check("wave: whole body at progress 1", fracs[-1] > 0.99, str(fracs))


def test_hat_lands_on_head():
    from voiceassistant.ui import mascot as po2
    import numpy as np
    base = po2.render_frame("specialist", "specialist", 1.0, None, 0.0, "idle", 0.3, 0.0)
    half = po2.render_frame("specialist", "specialist", 1.0, None, 0.5, "idle", 0.3, 0.0)
    full = po2.render_frame("specialist", "specialist", 1.0, None, 1.0, "idle", 0.3, 0.0)
    hat_px = lambda g: int(np.sum(np.all(g[..., :3] == np.array(po2.HAT), axis=2)))
    check("no hat before it is put on", hat_px(base) == 0, str(hat_px(base)))
    check("hat is visible mid-drop and once landed", 0 < hat_px(half) and 0 < hat_px(full), f"{hat_px(half)} {hat_px(full)}")
    assistant = po2.render_frame("assistant", "assistant", 1.0, None, 1.0, "idle", 0.3, 0.0)
    check("assistant never wears the hat", hat_px(assistant) == 0, str(hat_px(assistant)))


def test_mascot_switch_and_answer():
    """The widget: the colour starts to change at once, the hat only after the colour, and the box follows
    the answer's height and closes when the answer is cleared."""
    import os as _os
    import random as _r
    _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from voiceassistant.ui import mascot as po2
    from voiceassistant.ui import mascot as mm
    m = mm.Mascot(rng=_r.Random(1))
    check("mascot starts as assistant, no switch", m.agent == "assistant" and m._switch is None, "")
    m.set_agent("specialist")
    check("switch records assistant -> specialist", m._switch is not None and m._switch[0] == "assistant"
          and m._switch[1] == "specialist", str(m._switch))
    check("colour still assistant at the moment of switch", m._displayed_colour() == "assistant", "")
    m.t = po2.COLOUR_S
    st = m._state()
    check("after the colour wave: hat not on yet", st["hat"] == 0.0 and st["progress"] == 1.0, str(st))
    m.t = po2.COLOUR_S + po2.HAT_S + 0.01
    st = m._state()
    check("after the hat drop: hat on, switch finished", st["hat"] == 1.0 and m._switch is None, str(st))
    m.set_agent("assistant")
    m.t += 0.1
    st = m._state()
    check("back to assistant: hat comes off first", 0.0 < st["hat"] < 1.0 and st["progress"] == 0.0, str(st))
    check("idle: no box at all", m._turns == 0 and not m._recording and m.target_h == 0.0, str(m.target_h))
    m.add_question("What is on my screen?")
    m.set_answer("<b>x</b> " * 40)
    for _ in range(200):
        m._tick()
    check("box grew to the answer", 20 < m.box_h <= po2.BOX_MAX_H, str(m.box_h))
    m.add_question("And the next one?")
    check("a second question adds a turn, box still open", m._turns == 2 and m.target_h > 0, str(m._turns))
    idle = mm.Mascot(rng=_r.Random(2))
    for _ in range(200):
        idle._last_tick = None         # each call is one frame, as the timer delivers them
        idle._tick()
    check("idle mascot draws no box", idle.box_h < 1.0, str(idle.box_h))
    idle.set_recording(True)
    for _ in range(200):
        idle._last_tick = None
        idle._tick()
    check("recording opens the box for the bars", idle.box_h >= 20, str(idle.box_h))
    idle.push_level(80.0)
    check("levels are kept for the bars", idle._levels[-1] == 80.0, str(idle._levels[-1]))


# --- folder access and the name gate ----------------------------------------------------------------

def test_find_folders_and_permissions():
    """A folder is found by the name said (fuzzy), and the permission file lets the specialist write only in
    approved folders (and its own) and never run a shell."""
    import json as _json
    import tempfile as _tmp
    from voiceassistant.agents import specialist as sp
    root = _tmp.mkdtemp()
    os.makedirs(os.path.join(root, "projects", "example-outbox"))
    os.makedirs(os.path.join(root, ".hidden", "Outbox"))
    hits = sp.find_folders("example outbox", root=root)
    check("folder found by its name", len(hits) == 1 and hits[0].endswith("example-outbox"), str(hits))
    check("hidden folders are skipped", sp.find_folders("Outbox", root=root) == [], str(sp.find_folders("Outbox", root=root)))
    check("unrelated name finds nothing", sp.find_folders("Bananas", root=root) == [], "")
    real_dir = sp.SPECIALIST_DIR
    sp.SPECIALIST_DIR = _tmp.mkdtemp()
    try:
        approved = os.path.join(root, "projects", "example-outbox")
        sp.write_permissions([approved])
        with open(os.path.join(sp.SPECIALIST_DIR, "opencode.json")) as f:
            perm = _json.load(f)["permission"]
        check("approved folder may be read", perm["external_directory"].get(approved + "/**") == "allow", str(perm))
        check("approved folder is read-only", perm["edit"].get(approved + "/**") == "deny", str(perm))
        check("the specialist's own folder is writable", perm["edit"].get("*") == "allow", str(perm))
        keys = list(perm["edit"])
        check("the approved folder's deny comes before the catch-all allow",
              keys.index(approved + "/**") < keys.index("*"), str(perm["edit"]))
        check("no shell", perm["bash"] == {"*": "deny"}, str(perm["bash"]))
    finally:
        sp.SPECIALIST_DIR = real_dir


def test_name_gate():
    """A turn counts only when it starts with the name; with no name set everything passes."""
    from voiceassistant.core import config as cfgm
    cfg = {"assistant": {"name": "Octávio", "wake_variants": ["octavi"]}}
    check("starts with the name", cfgm.addressed_to("Octávio, que horas são?", cfg), "")
    check("name variant heard by setup", cfgm.addressed_to("Octavi, para", cfg), "")
    check("random speech is not addressed", not cfgm.addressed_to("Thank you.", cfg), "")
    check("name in the middle does not count", not cfgm.addressed_to("Que horas são, Octávio?", cfg), "")
    check("no name set: nothing is gated", cfgm.addressed_to("Thank you.", {"assistant": {"name": None}}), "")


def test_setup_name_forms():
    """Setup keeps the first word Whisper wrote for the name, and only where it is the name."""
    from voiceassistant.core import config as cfgm
    takes = ["Zeta, que horas são?", "Zeta, para.", "Seta, qual é a capital?", "Carla, o que está no ecrã"]
    check("a name two takes agree on is kept", "zeta" in cfgm.name_forms_from_takes(takes),
          str(cfgm.name_forms_from_takes(takes)))
    check("a one-off mishearing is not kept", "seta" not in cfgm.name_forms_from_takes(takes), "")
    check("a name the mic writes one way is learnt (Todd for Thoth)",
          cfgm.name_forms_from_takes(["Todd, desenha um diagrama.", "Todd, explica o router.", "Thank you."]) == ["todd"],
          "")


def test_addressee():
    """A turn goes to the assistant (Zeta), to the specialist (Thoth), or to nobody (random speech)."""
    from voiceassistant.core import config as cfgm
    cfg = {"assistant": {"name": "Zeta", "wake_variants": ["seta"]},
           "providers": {"specialist": {"name": "Thoth"}}}
    check("Zeta, ... goes to the assistant", cfgm.addressee("Zeta, que horas são?", cfg) == "assistant", "")
    check("a form Zeta was heard as goes to the assistant", cfgm.addressee("Seta, para.", cfg) == "assistant", "")
    check("Thoth, ... goes to the specialist", cfgm.addressee("Thoth, lê a pasta.", cfg) == "specialist", "")
    check("random speech goes to nobody", cfgm.addressee("Thank you.", cfg) is None, "")
    check("the name in the middle does not count", cfgm.addressee("Que horas são, Zeta?", cfg) is None, "")
    check("no names set: everything is for the assistant",
          cfgm.addressee("Thank you.", {"assistant": {"name": None}, "providers": {"specialist": {"name": None}}})
          == "assistant", "")
    named = {"assistant": {"name": "Zeta", "wake_variants": ["seta"]},
             "providers": {"specialist": {"name": "vulk", "wake_variants": ["valk", "volk"]}}}
    check("Volk, ... goes to the specialist", cfgm.addressee("Volk, lê a pasta.", named) == "specialist", "")
    check("Valk, ... goes to the specialist", cfgm.addressee("Valk, lê a pasta.", named) == "specialist", "")
    check("Zeta, ... beside the specialist's forms goes to the assistant",
          cfgm.addressee("Zeta, que horas?", named) == "assistant", "")
    check("a near-miss of Zeta still goes to the assistant", cfgm.addressee("Seta, para.", named) == "assistant", "")
    check("a first letter Whisper wrote (Z) goes to the assistant", cfgm.addressee("Z, qual o tempo", named) == "assistant", "")
    check("a first letter Whisper wrote (V) goes to the specialist", cfgm.addressee("V, lê a pasta.", named) == "specialist", "")
    check("a first letter that is no name's start goes to nobody", cfgm.addressee("B, qual o tempo", named) is None, "")



def test_answer_language_guard():
    """An answer in French is not spoken; English and European Portuguese pass; short text is not judged."""
    from voiceassistant import app as vo
    french = "Bonjour, je voudrais savoir comment fonctionne le routeur de la maison aujourd hui."
    portuguese = "Olá, o router de casa está ligado à rede da sala e funciona normalmente hoje."
    english = "Hello, the router in the house is connected to the network and works normally today."
    check("guard: French answer is not accepted", not vo._answer_in_language(french), "")
    check("guard: European Portuguese answer is accepted", vo._answer_in_language(portuguese), "")
    check("guard: English answer is accepted", vo._answer_in_language(english), "")
    check("guard: a short answer is not judged", vo._answer_in_language("Bonjour."), "")

def test_open_tool():
    """An open request names a folder; only approved folders (and the specialist's own) can be opened."""
    import tempfile as _tmp
    from voiceassistant.agents import open_tool as ot
    check("abre a pasta Outbox -> Outbox", ot.open_name("Abre a pasta Outbox.") == "Outbox", str(ot.open_name("Abre a pasta Outbox.")))
    check("open the Outbox folder -> Outbox", ot.open_name("Open the Outbox folder") == "Outbox", "")
    check("a question is not an open request", ot.open_name("Que horas são?") is None, "")
    approved = _tmp.mkdtemp()
    cfg = {"access": {"folders": [approved]}}
    check("a file inside an approved folder may be opened", ot.is_allowed(os.path.join(approved, "x.txt"), cfg), "")
    check("a folder outside every approved folder may not", not ot.is_allowed("/etc/passwd", cfg), "")
    link = os.path.join(approved, "escape")
    os.symlink("/etc", link)
    check("a link out of an approved folder is refused", not ot.is_allowed(os.path.join(link, "passwd"), cfg), "")


def test_open_apps():
    """An app is found by its desktop name; hidden entries are skipped; a near name still finds it."""
    import tempfile
    from voiceassistant.agents import open_tool as ot
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "firefox.desktop"), "w") as f:
            f.write("[Desktop Entry]\nType=Application\nName=Firefox\nExec=firefox %u\n")
        with open(os.path.join(d, "hidden.desktop"), "w") as f:
            f.write("[Desktop Entry]\nType=Application\nName=Secret\nNoDisplay=true\n")
        saved = ot.DESKTOP_DIRS
        ot.DESKTOP_DIRS = [d]
        try:
            check("app: found by name", (ot.find_app("firefox") or {}).get("id") == "firefox", "")
            check("app: a near name is found", (ot.find_app("Firefx") or {}).get("id") == "firefox", "")
            check("app: hidden entry is not an app", ot.find_app("Secret") is None, "")
            check("app: an unknown name finds nothing", ot.find_app("Banana") is None, "")
        finally:
            ot.DESKTOP_DIRS = saved


def test_herdr_tools():
    """The front desk's pane tools: a question about panes offers them, listing and reading run without changes,
    and a send is only staged (the caller reads it back and waits for the user's yes)."""
    from voiceassistant.agents import herdr_pane as hp
    from voiceassistant.answers import realtime_answer as ra
    check("a question about panes offers the pane tools", ra.needs_panes("Lê o painel Zeta"), "")
    check("a question without panes does not", not ra.needs_panes("Que horas são em Lisboa?"), "")
    check("a send names the pane and the text", hp.parse_request("Envia ao painel vulk: corre os testes")
          == ("send", {"pane": "vulk", "text": "corre os testes"}), str(hp.parse_request("Envia ao painel vulk: corre os testes")))
    check("a send can say 'que' before the text", hp.parse_request("manda para o painel vulk que corre os testes")
          == ("send", {"pane": "vulk", "text": "corre os testes"}), "")
    check("a read names the pane", hp.parse_request("O que está no painel Zeta?") == ("read", {"pane": "Zeta"}), "")
    check("a list asks for the open panes", hp.parse_request("Quais são os painéis abertos?") == ("list", {}), "")
    check("a question that is not about panes parses to nothing", hp.parse_request("Que horas são em Lisboa?") is None, "")
    saved_list, saved_read = hp.list_panes, hp.read_pane
    hp.list_panes = lambda: [{"id": "p1", "title": "vulk", "cwd": "/work/proj", "focused": False, "agent": "idle"}]
    hp.read_pane = lambda pid, lines=60: "linha 1\nlinha 2"
    try:
        check("list_panes lists title and folder", '"vulk"' in hp.run_tool("list_panes", {}), "")
        check("read_pane reads the one matching pane", "linha 2" in hp.run_tool("read_pane", {"pane": "vulk"}), "")
        out = hp.run_tool("send_to_pane", {"pane": "vulk", "text": "corre os testes"})
        check("send_to_pane only stages, and says so", out.startswith("Preparado"), out)
        check("the staged prompt waits for the user's yes", hp.pending() is not None
              and hp.pending()["text"] == "corre os testes", str(hp.pending()))
        hp.cancel()
        check("no matching pane gives a plain message", "Não encontrei" in hp.run_tool("read_pane", {"pane": "nada"}), "")
    finally:
        hp.list_panes, hp.read_pane = saved_list, saved_read
        hp.cancel()


def test_camera_pose():
    """The camera is a pose of its own: it parses, it flashes at the start of each shutter cycle, and it is off between."""
    from voiceassistant.ui import mascot as mm
    check("camera weight is read from the pose dict",
          mm._parse_poses({"search": 0.0, "write": 0.0, "camera": 0.7}) == (0.0, 0.0, 0.7), "")
    check("the camera is a blended pose like search and write", "camera" in mm.POSES, str(mm.POSES))
    check("the shutter flashes at the start of a cycle", mm._shutter_flash(0.02) > 0.5, "")
    check("no flash between shutter clicks", mm._shutter_flash(1.0) == 0.0, "")


FAILED: list[str] = []
PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
    else:
        FAILED.append(f"{name}: {detail}")


# --- transcripts: what counts as noise --------------------------------------------------------------

def test_noise_transcripts():
    check("empty is noise", vo.is_noise_transcript(""), "")
    check("dots are noise", vo.is_noise_transcript("..."), "")
    check("subtitle hallucination is noise", vo.is_noise_transcript("Thank you."), "")
    check("foreign-script hallucination is noise", vo.is_noise_transcript("Продолжение следует..."), "")
    check("real question is not noise", not vo.is_noise_transcript("What is the capital of Portugal?"), "")
    check("Portuguese question is not noise", not vo.is_noise_transcript("Quantos habitantes tem Lisboa?"), "")


# --- stop requests ----------------------------------------------------------------------------------

def test_stop_requests():
    for phrase in ["Stop.", "stop", "Please stop now", "STOP"]:
        check(f"stop: {phrase}", is_stop_request(phrase), phrase)
    for phrase in ["Para a conversa.", "Chega.", "Goodbye", "That's all", "That is all", "never mind",
                   "What is the capital of Portugal?", "Quantos habitantes tem Lisboa?", "Obrigado.",
                   "Thank you.", "Você pode parar a música?"]:
        check(f"not stop: {phrase}", not is_stop_request(phrase), phrase)


# --- echo check during an answer --------------------------------------------------------------------

def test_echo_check():
    spoken = "The capital of Portugal is Lisbon."
    check("echo of the answer is ignored", vo._is_echo("The capital of Portugal", spoken), "")
    check("user talking over the answer is not echo",
          not vo._is_echo("Stop, what is the population of Porto", spoken), "")
    check("nothing intelligible counts as echo", vo._is_echo("", spoken), "")
    check("noise heard over the answer (thank you) is not talking over it", vo._is_echo("Thank you.", spoken), "")
    check("Russian keyboard noise over the answer is not talking over it", vo._is_echo("Спасибо", spoken), "")


# --- endpointing on synthetic frames ----------------------------------------------------------------

def frames(rms: float, n: int) -> list[float]:
    return [rms] * n


def run_endpointer(levels: list[float]) -> tuple[int | None, int | None]:
    """Returns the frame index of onset and of end, or None."""
    ep = vo.Endpointer(1000)
    onset = end = None
    for i, rms in enumerate(levels):
        event = ep.update(rms)
        if event == "onset" and onset is None:
            onset = i
        if event == "end" and end is None:
            end = i
            break
    return onset, end


def test_endpointer_basic():
    levels = frames(5, vo.WARMUP_FRAMES + 5) + frames(2000, 40) + frames(10, 40)
    onset, end = run_endpointer(levels)
    check("basic: onset found", onset is not None, f"onset={onset}")
    check("basic: end found after speech", end is not None and end > (onset or 0), f"end={end}")


def test_endpointer_ignores_click():
    # a loud keypress click in the first 0.6 s must not open a turn
    levels = [18000] * 6 + frames(5, vo.WARMUP_FRAMES) + frames(5, 30)
    onset, _ = run_endpointer(levels)
    check("click: no onset from the keypress", onset is None, f"onset={onset}")


def test_endpointer_recovers_from_loud_start():
    # A loud sound just after the shortcut opens a turn, which ends at the quiet gap; the recorder then
    # starts a new turn (a new Endpointer) for the question that follows. The new turn must hear it.
    burst_turn = frames(5, vo.WARMUP_FRAMES) + frames(18000, 20) + frames(10, 40)
    onset1, end1 = run_endpointer(burst_turn)
    check("loud start: the burst opens a turn", onset1 is not None, f"onset={onset1}")
    check("loud start: the turn ends at the gap after it", end1 is not None, f"end={end1}")
    question_turn = frames(5, vo.WARMUP_FRAMES) + frames(3000, 40) + frames(10, 40)
    onset2, end2 = run_endpointer(question_turn)
    check("loud start: the question is heard in the next turn", onset2 is not None and end2 is not None,
          f"onset={onset2} end={end2}")


def test_endpointer_no_end_during_short_pause():
    levels = frames(5, vo.WARMUP_FRAMES) + frames(2000, 40) + frames(10, 5) + frames(2000, 40) + frames(10, 40)
    onset, end = run_endpointer(levels)
    check("short pause does not end the turn", end is not None and end >= vo.WARMUP_FRAMES + 40 + 5 + 40,
          f"end={end}")


# --- text for the model and the voice ---------------------------------------------------------------

def test_history_prompt():
    check("no history: the question alone", _with_history("Q?", []) == "Q?", "")
    p = _with_history("Nova?", [("Primeira?", "Resposta.")])
    check("history: earlier turn is included", "Primeira?" in p and "Resposta." in p, p)
    check("history: new question last", p.rstrip().endswith("Nova?"), p)


def test_speech_text_cleaning():
    s = "Aqui está: <html><body>teste</body></html> e ```python\nprint(1)\n``` **Lisboa** é a capital $x^2$."
    out = plain(s)
    check("speech: no HTML", "<" not in out and "html" not in out, out)
    check("speech: no code", "print" not in out, out)
    check("speech: no bold markers", "*" not in out, out)
    check("speech: inline code is spoken", "config.yaml" in plain("Usa `config.yaml` aqui."), "")
    check("speech: a link is read as its words", plain("Vê [o site](https://x.y/z) já.") == "Vê o site já.", plain("Vê [o site](https://x.y/z) já."))
    check("speech: a table is read as cells", plain("| A | B |\n|---|---|\n| 1 | 2 |") == "A, B. 1, 2.", plain("| A | B |\n|---|---|\n| 1 | 2 |"))
    check("speech: a tilde in a path is kept", "~/.config" in plain("Vê `~/.config` aqui.") or "/.config" in plain("Vê `~/.config` aqui."), "")
    check("speech: keeps the words", "Lisboa" in out and "capital" in out, out)


def test_search_routing():
    check("search: release date question", needs_search("When was Unreal 5.9 released?"), "")
    check("search: Portuguese population", needs_search("Quantos habitantes tem Lisboa?"), "")
    check("no search: general knowledge", not needs_search("Say hello"), "")


# --- screen questions -------------------------------------------------------------------------------

def test_screen_routing():
    check("screen: English", needs_screen("What is on my left screen?"), "")
    check("screen: Portuguese", needs_screen("O que está no meu ecrã da direita?"), "")
    check("screen: not a screen question", not needs_screen("What is the capital of Portugal?"), "")
    check("side: right", pick_side("What is on my right screen?") == "right", "")
    check("side: direita", pick_side("O que está no ecrã da direita?") == "right", "")
    check("side: left", pick_side("What is on my left screen?") == "left", "")
    check("side: default left", pick_side("What is on my screen?") == "left", "")


# --- search results ---------------------------------------------------------------------------------

def test_search_filter_and_parse():
    check("foreign share: Chinese text", web_search._foreign_share("中文测试") > 0.9, "")
    check("foreign share: Portuguese", web_search._foreign_share("Lisboa é a capital") == 0.0, "")
    real = web_search.search_url("Unreal 5.9")
    check("search url uses the herdr parameters", "format=json" in real and "language=auto" in real, real)


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
        except Exception as e:
            FAILED.append(f"{t.__name__} raised {type(e).__name__}: {e}")
    total = PASSED + len(FAILED)
    for f in FAILED:
        print("FAIL", f)
    print(f"{PASSED}/{total} checks passed")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
