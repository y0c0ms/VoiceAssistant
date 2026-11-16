"""screen.py — questions about what is on a screen.

The desktop is captured with spectacle, the chosen monitor is cropped out, and LFM3-Omni answers from the
image (llama-swap, OpenAI chat API with an image part).

Monitors (kscreen-doctor): DP-2 at 0,0 is the left and primary screen; HDMI-A-3 at 1920,0 is the right one.
"""
import base64
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request


def _chat_url() -> str:
    """VA_CHAT (a full URL) if set, else the front desk's address from the config file."""
    if os.environ.get("VA_CHAT"):
        return os.environ["VA_CHAT"]
    from voiceassistant.core import config
    return config.service_url("front_desk") + "/v1/chat/completions"

LLM_MODEL = os.environ.get("VA_MODEL", "lfm3-omni")
MON_W, MON_H = 1920, 1080
MONITOR_X = {"left": 0, "right": 1920}

SCREEN_RE = re.compile(r"\b(ecr[ãa]|ecran|écran|screen|tela|monitor|janela|window)\b", re.IGNORECASE)
LEFT_RE = re.compile(r"\b(esquerd[oa]s?|left)\b", re.IGNORECASE)
RIGHT_RE = re.compile(r"\b(direit[oa]s?|right)\b", re.IGNORECASE)


def needs_screen(question: str) -> bool:
    return bool(SCREEN_RE.search(question))


def pick_side(question: str) -> str:
    """'left' or 'right' from the question; the primary (left) screen when it does not say."""
    if RIGHT_RE.search(question) and not LEFT_RE.search(question):
        return "right"
    return "left"


def capture_jpeg(side: str) -> bytes:
    """JPEG of one monitor, 1280 px wide. The full desktop is captured once and cropped."""
    full = os.path.join(tempfile.gettempdir(), f"va_screen_{time.time_ns()}.png")
    subprocess.run(["spectacle", "-b", "-f", "-o", full], check=True, capture_output=True, timeout=30)
    crop = f"{MONITOR_X[side]}"
    out = full[:-4] + f"_{side}.jpg"
    subprocess.run(["magick", full, "-crop", f"{MON_W}x{MON_H}+{crop}+0", "+repage",
                    "-resize", "1280x", "-quality", "80", out], check=True, capture_output=True, timeout=30)
    with open(out, "rb") as f:
        data = f.read()
    os.remove(full)
    return data


def ask_screen(question: str, side: str | None = None) -> str:
    """The model's answer about the chosen monitor. Blocking: run it in a thread."""
    side = side or pick_side(question)
    jpeg = capture_jpeg(side)
    b64 = base64.b64encode(jpeg).decode()
    prompt = (question + " Answer in the language of the question, briefly, from what is visible on the screen. "
              "If something is not readable, say so.")
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 300,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(_chat_url(), data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"]["content"].strip()
