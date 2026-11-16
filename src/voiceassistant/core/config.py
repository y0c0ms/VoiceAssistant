"""config.py: the one configuration file of the voice assistant, YAML.

The file is ~/.config/voiceassistant/config.yaml. A project ships config.example.yaml; the first run copies it
there if there is no config yet. Missing keys fall back to DEFAULTS, so a partial file is fine.

Sections:
  assistant  the name a turn must start with, what setup heard of it, whether setup was done
  tts        voice and speed per language (en: LFM3 voice; pt: dii or miro)
  providers  the specialist (opencode or qwen-local), the front desk model, transcription, search
  mic        which microphone source to use (null: the echo-cancelled one, or the default)
"""
import copy
import difflib
import os
import re
import shutil
import unicodedata

import yaml
from voiceassistant.paths import EXAMPLE_CONFIG

PATH = os.path.expanduser("~/.config/voiceassistant/config.yaml")
EXAMPLE = EXAMPLE_CONFIG

DEFAULTS = {
    "assistant": {"name": None, "wake_variants": [], "wake_level": None, "setup_done": False},
    "tts": {
        "en": {"voice": "female", "tempo": 1.0},
        "pt": {"voice": "dii", "tempo": 1.0},
    },
    "providers": {
        "specialist": {
            "name": "vulk",                                       # what the user calls it to address it
            "wake_variants": [],                                   # the forms its setup heard (e.g. "todd")
            "setup_done": False,
            "backend": "opencode",                                 # opencode | qwen-local
            "opencode_model": "opencode/deepseek-v4.1-flash-free",
            "opencode_agent": "build",
            "opencode_dir": "~/.local/share/voiceassistant/specialist",
            "local_model": "qwen3.6-35b",
        },
        "front_desk": {"url": "http://127.0.0.1:11435", "model": "lfm3-omni"},
        "transcription": {"url": "http://127.0.0.1:11435/v1/audio/transcriptions", "model": "whisper-gpu"},
        "search": {"url": "http://127.0.0.1:30850"},
        "router": {"url": "http://127.0.0.1:11441/v1/systemone"},   # System One: CPU option scorer
    },
    "mic": {"source": None},
    "access": {"folders": []},     # folders the specialist may read and write; added by voice consent
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str = PATH) -> dict:
    """The configuration, with defaults for whatever the file does not say."""
    try:
        with open(path) as f:
            saved = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        saved = {}
    return _merge(DEFAULTS, saved)


def service_url(key: str, env: str | None = None) -> str:
    """The address of a model service: the environment variable when it is set (env), otherwise the address in the
    config file (providers.<key>.url). The config file is where a machine says where its model server runs."""
    if env and os.environ.get(env):
        return os.environ[env]
    return load()["providers"][key]["url"]


def save(cfg: dict, path: str = PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def ensure_file(path: str = PATH) -> str:
    """First run: copy the shipped example to the user's config, if there is none yet."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copyfile(EXAMPLE, path)
    return path


# --- helpers used by the modules ----------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lower case, no accents, no punctuation: 'Octávio,' -> 'octavio'."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", plain).strip()


def first_word(text: str) -> str:
    words = normalize(text).split()
    return words[0] if words else ""


def name_forms_from_takes(takes: list[str], min_takes: int = 2) -> list[str]:
    """The first words Whisper wrote for the name across the setup takes. A form counts once at least `min_takes` takes
    agree on it, so a one-off mishearing does not become a name, while a name the mic always writes the same way
    ("Todd" for Thoth) does. Words under three letters are not names."""
    counts: dict[str, int] = {}
    for text in takes:
        w = first_word(text)
        if len(w) >= 3:
            counts[w] = counts.get(w, 0) + 1
    return [w for w, c in counts.items() if c >= min_takes]


def _forms(name: str | None, variants: list[str] | None) -> list[str]:
    return ([normalize(name)] if name else []) + list(variants or [])


MATCH_MIN_SHORT = 0.75   # names of four letters or fewer: one letter wrong is still a match ("seta" for "zeta")
MATCH_MIN_LONG = 0.8     # longer names may differ by a letter or two


def _name_score(word: str, forms: list[str]) -> float:
    """How close the heard word is to an agent's forms: 1.0 exact, the best similarity above the match floor, else 0."""
    best = 0.0
    for form in forms:
        if not word or not form:
            continue
        if word == form:
            return 1.0
        if len(word) == 1 and form.startswith(word):
            best = max(best, MATCH_MIN_SHORT)          # Whisper often writes only the first letter ("Z" for Zeta)
            continue
        floor = MATCH_MIN_SHORT if min(len(word), len(form)) <= 4 else MATCH_MIN_LONG
        ratio = difflib.SequenceMatcher(None, word, form).ratio()
        if ratio >= floor:
            best = max(best, ratio)
    return best


def addressed_to(text: str, cfg: dict) -> bool:
    """True when the turn starts with the assistant's name or a form its setup heard. No name set: always True."""
    a = cfg["assistant"]
    if not a.get("name"):
        return True
    return addressee(text, cfg) == "assistant"


def addressee(text: str, cfg: dict) -> str | None:
    """Who a turn is for: the agent whose name or heard form is nearest the first word, if any is near enough.
    A tie or no near name gives None, so a turn that is unclear is not sent to either agent. With no names set,
    everything is for the assistant."""
    a, s = cfg["assistant"], cfg.get("providers", {}).get("specialist", {})
    if not a.get("name") and not s.get("name"):
        return "assistant"
    heard = first_word(text)
    sa = _name_score(heard, _forms(a.get("name"), a.get("wake_variants")))
    ss = _name_score(heard, _forms(s.get("name"), s.get("wake_variants")))
    if sa == 0.0 and ss == 0.0:
        return None
    if ss > sa:
        return "specialist"
    if sa > ss:
        return "assistant"
    return None
