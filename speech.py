"""speech.py — speak text in the right voice, per sentence.

- Portuguese sentences: Phoonnx dii tugaphone (local, pt-PT), via .venv-phoonnx.
- English sentences: the LFM3-Omni female voice (llama-server /tts), raised to dii's pitch and
  slowed to 85% with ffmpeg (asetrate + atempo) so the two voices sound like one speaker.
- Consecutive sentences of one language form one segment; a language change starts the next
  segment, so mixed text switches voice mid-answer.
- speak(..., cancel=event) stops the playback as soon as the event is set.

Language is detected with langid restricted to pt and en.
"""
import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

# langid is imported lazily, see _langid()

import config
from audio_devices import playback_sink
from rich_text import plain
from voice_chat import resolve_realtime_url

_LANGID = None


def _langid():
    """langid is loaded on first use: its setup costs most of a second at import time."""
    global _LANGID
    if _LANGID is None:
        import langid
        langid.set_languages(["pt", "en"])
        _LANGID = langid
    return _LANGID

HERE = os.path.dirname(os.path.abspath(__file__))
PHOONNX_PYTHON = os.path.join(HERE, ".venv-phoonnx", "bin", "python")
TUGAPHONE_SCRIPT = os.path.join(HERE, "tts_tugaphone.py")

# LFM3 "female" median F0 is 186 Hz, dii's is 208 Hz: raise the English voice by that gap.
EN_PITCH_RATIO = 208.0 / 186.0

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def _tts(lang: str) -> dict:
    """The voice and tempo of a language, as the config file says (read again for each sentence)."""
    return config.load()["tts"][lang]


def _en_filter(tempo: float) -> str:
    """Pitch up by EN_PITCH_RATIO and keep the tempo, then set the tempo. LFM3 audio is 24 kHz."""
    r = EN_PITCH_RATIO
    return f"asetrate=24000*{r:.6f},aresample=24000,atempo={1 / r:.6f},atempo={tempo}"


def lang_of(text: str) -> str:
    """'en' or 'pt' for a piece of text."""
    return _langid().classify(text)[0]


_ANY_LANG = None


def guess_language(text: str) -> str:
    """Any language langid knows, not only pt and en: the answer guard has to see a French answer as French."""
    global _ANY_LANG
    if _ANY_LANG is None:
        from langid.langid import LanguageIdentifier, model
        _ANY_LANG = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    return _ANY_LANG.classify(text)[0]


def segments(text: str) -> list[tuple[str, str]]:
    """Split text into (lang, sentence) pairs, one per sentence. Each sentence is synthesised and played on its own,
    so the first one starts as soon as it is ready instead of after the whole answer is synthesised."""
    out: list[tuple[str, str]] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if sentence:
            out.append((lang_of(sentence), sentence))
    return out


def _synth_pt(text: str, path: str) -> None:
    pt = _tts("pt")
    subprocess.run([PHOONNX_PYTHON, TUGAPHONE_SCRIPT, text, "-o", path, "--voice", pt["voice"]],
                   check=True, capture_output=True, timeout=60)
    if pt["tempo"] != 1.0:
        tmp = path + ".tempo.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", path,
                        "-filter:a", f"atempo={pt['tempo']}", tmp], check=True, timeout=30)
        os.replace(tmp, path)


def _synth_en(text: str, path: str) -> None:
    base = urllib.parse.urlsplit(resolve_realtime_url())
    body = json.dumps({"input": text, "voice": _tts("en")["voice"]}).encode()
    req = urllib.request.Request(f"http://{base.hostname}:{base.port}/tts", data=body,
                                 headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    raw_path = path + ".raw.wav"
    try:
        with open(raw_path, "wb") as f:
            f.write(raw)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw_path,
                        "-filter:a", _en_filter(_tts("en")["tempo"]), path], check=True, timeout=30)
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)


async def _play(path: str, cancel: asyncio.Event | None) -> None:
    """Play one file; returns early, killing the player, if `cancel` is set."""
    sink = playback_sink()
    args = ["paplay"] + ([f"--device={sink}"] if sink else []) + [path]
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    if cancel is None:
        await proc.wait()
        return
    done = asyncio.create_task(proc.wait())
    stopped = asyncio.create_task(cancel.wait())
    try:
        await asyncio.wait({done, stopped}, return_when=asyncio.FIRST_COMPLETED)
        if stopped.done() and not done.done():
            print(f"tts stopped early by cancel: {os.path.basename(path)}", flush=True)
    finally:
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()
        for t in (done, stopped):
            t.cancel()


async def speak(text: str, cancel: asyncio.Event | None = None) -> None:
    """Speak text one sentence at a time, switching voice at each language change. The next sentence is synthesised
    while the current one plays; stops early when `cancel` is set."""
    items = segments(plain(text))

    def start(i: int):
        lang, sentence = items[i]
        path = os.path.join(tempfile.gettempdir(), f"va_speech_{time.time_ns()}.wav")
        synth = _synth_en if lang == "en" else _synth_pt
        return path, asyncio.create_task(asyncio.to_thread(synth, sentence, path))

    def discard(path: str, task: asyncio.Task) -> None:
        # a sentence that will not play: its file goes once its synthesis has finished
        def remove(_=None):
            if os.path.exists(path):
                os.remove(path)
        if task.done():
            remove()
        else:
            task.add_done_callback(remove)

    pending = start(0) if items else None
    current = None
    try:
        for i in range(len(items)):
            path, task = pending
            pending = None
            current = (path, task)
            await task
            if cancel is not None and cancel.is_set():
                return
            if i + 1 < len(items):
                pending = start(i + 1)
            await _play(path, cancel)
            if os.path.exists(path):
                os.remove(path)
            current = None
            if cancel is not None and cancel.is_set():
                return
    finally:
        if current is not None:
            discard(*current)
        if pending is not None:
            discard(*pending)
