"""open_tool.py: open a folder, a diagram file, or an installed app for the user, and nothing else.

Only paths inside the approved folders (config access.folders) or the specialist's own folder can be opened. A
path is resolved first (symlinks too), so a link out of an approved folder is refused. Folders and files open with
`xdg-open`, apps with `gtk-launch` on their desktop id; both run detached, with no shell and no arguments from the request.
"""
import difflib
import os
import re
import subprocess

from voiceassistant.core import config
from voiceassistant.agents import specialist

OPEN_RE = re.compile(
    r"^\s*(?:abre|abrir|abra|open)\s+(?:(?:a|o|the)\s+)?(?:(?:pasta|folder|ficheiro|file)\s+)?(?P<name>.+?)"
    r"(?:\s+(?:folder|pasta|ficheiro|file))?\s*[?.!]*\s*$",
    re.IGNORECASE)


def open_name(text: str) -> str | None:
    """The thing the user asked to open, when the request is an open request; otherwise None."""
    m = OPEN_RE.match(text)
    if not m:
        return None
    name = m.group("name").strip().strip("\"'")
    return name or None


def allowed_roots(cfg: dict | None = None) -> list[str]:
    cfg = cfg or config.load()
    roots = [os.path.realpath(os.path.expanduser(f)) for f in cfg["access"]["folders"]]
    roots.append(os.path.realpath(specialist.SPECIALIST_DIR))
    return roots


def is_allowed(path: str, cfg: dict | None = None) -> bool:
    real = os.path.realpath(os.path.expanduser(path))
    return any(real == r or real.startswith(r + os.sep) for r in allowed_roots(cfg))


def open_path(path: str) -> bool:
    """Open the path with the desktop's default app. False when it is outside the approved folders."""
    if not is_allowed(path) or not os.path.exists(path):
        return False
    subprocess.Popen(["xdg-open", os.path.realpath(path)], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return True


def approved_folders_named(name: str) -> list[str]:
    """Folders with that name under the approved folders (the specialist's own folder is not a user folder)."""
    hits: list[str] = []
    for root in config.load()["access"]["folders"]:
        base = os.path.realpath(os.path.expanduser(root))
        hits += [h for h in specialist.find_folders(name, root=base) if is_allowed(h)]
        if os.path.basename(base).lower() == name.lower() or config.normalize(os.path.basename(base)) == config.normalize(name):
            hits.append(base)
    return list(dict.fromkeys(hits))


def diagram_files(answer: str) -> list[str]:
    """HTML files the specialist named in its answer, that exist inside the specialist's folder."""
    found = re.findall(r"[\w.\-]+\.html", answer)
    out = []
    for f in found:
        path = os.path.join(specialist.SPECIALIST_DIR, f)
        if os.path.exists(path) and is_allowed(path):
            out.append(path)
    return list(dict.fromkeys(out))


DESKTOP_DIRS = ["/usr/share/applications", "/usr/local/share/applications",
                os.path.expanduser("~/.local/share/applications")]    # the user's own entries come last and win


def _desktop_entry(path: str) -> dict | None:
    """Name, and whether the entry is a launchable application, from a .desktop file's [Desktop Entry] section."""
    fields: dict[str, str] = {}
    in_entry = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                in_entry = line == "[Desktop Entry]"
                continue
            if in_entry and "=" in line:
                key, _, value = line.partition("=")
                fields.setdefault(key.strip(), value.strip())
    if fields.get("Type") != "Application" or fields.get("Hidden") == "true" or fields.get("NoDisplay") == "true":
        return None
    if "Name" not in fields:
        return None
    return {"id": os.path.basename(path)[:-len(".desktop")], "name": fields["Name"]}


def installed_apps() -> list[dict]:
    """Launchable applications from the desktop entries, by desktop id (the file name without .desktop)."""
    apps: dict[str, dict] = {}
    for d in DESKTOP_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".desktop"):
                continue
            entry = _desktop_entry(os.path.join(d, fn))
            if entry:
                apps[entry["id"]] = entry
    return list(apps.values())


def find_app(name: str) -> dict | None:
    """The installed application whose name is the one said (accents and case ignored; a letter or two may differ
    for longer names, as with the folders). None when no app is near enough."""
    want = config.normalize(name)
    best, best_score = None, 0.0
    for app in installed_apps():
        got = config.normalize(app["name"])
        if got == want:
            return app
        if len(want) >= 4 and len(got) >= 4:
            score = difflib.SequenceMatcher(None, got, want).ratio()
            if score >= 0.8 and score > best_score:
                best, best_score = app, score
    return best


def launch_app(app_id: str) -> bool:
    """Start an installed application by its desktop id, detached. No arguments from the request."""
    subprocess.Popen(["gtk-launch", app_id], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return True
