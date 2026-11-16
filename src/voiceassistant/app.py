#!/usr/bin/env python3
"""voice_overlay.py — voice assistant card: mic waveform, transcript, question and answer bubbles,
sources, and a conversation that stays open for follow-up turns.

Bind Shift+Super+Space to `voice_overlay.py --toggle`:
  first press  -> card opens bottom-right on the primary screen, the mic starts
  speaking     -> the turn ends by itself after a pause
  second press -> stops at once
Each turn is transcribed by Whisper (the dictation model, with language detection), answered by
LFM3-Omni, and spoken per sentence in the matching voice. Saying "stop" while the answer is spoken
cancels the playback. The card waits 25 s for the next question, then fades out. Esc closes it now.

Endpointing: 50 ms frames, 150 ms onset, the panel's silence window, 900 ms minimum speech, 15 s
cap. Speech and silence thresholds follow the room's own noise floor (see Endpointer).
"""
import argparse
import asyncio
import base64
import collections
import html
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import unicodedata
import wave

import numpy as np
import websockets
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication

from voiceassistant.core import config
from voiceassistant.speech import speech
from voiceassistant.answers.realtime_answer import answer
from voiceassistant.answers.screen import ask_screen, needs_screen
from voiceassistant.agents import specialist as specialist_mod
from voiceassistant.agents.specialist import Session as SpecialistSession, answer as specialist_answer, decide as specialist_decide
from voiceassistant.core.stop import is_stop_request
from voiceassistant.core import audio_devices
from voiceassistant.agents import herdr_pane
from voiceassistant.agents import open_tool
from voiceassistant.ui.overlay_window import Overlay
from voiceassistant.ui.tray import make_tray
from voiceassistant.core.voice_chat import resolve_realtime_url, write_wav

RATE = 16000
FRAME_MS = 50                              # VAD frame, as in herdr-desk-deck
FRAME_BYTES = RATE * 2 * FRAME_MS // 1000  # PCM16 mono
ONSET_FRAMES = 3                           # 150 ms of speech opens the turn
WARMUP_FRAMES = 12                         # 600 ms after the shortcut: the keypress click is not a question
PRE_ROLL_FRAMES = 8                        # 400 ms kept before the onset, so the first word is not cut
MIN_SPEECH_MS = 600                        # silence cannot end a shorter turn; a one-word "stop" is ~700 ms, a click is shorter
MAX_TURN_MS = 15000                        # hard cap on one turn
NO_SPEECH_MS = 10000                       # the first turn: give up if nothing is said
FOLLOWUP_MS = 25000                        # after an answer the card waits this long for the next turn
HEARD_ATTEMPTS = 3                         # a turn that transcribes to noise is heard again, this many times
HISTORY_TURNS = 4                          # earlier turns given to the model as context
FADE_DELAY_MS = 2500                       # the answer stays readable this long after it ends
FADE_MS = 900                              # then the card fades out over this long
DEFAULT_SILENCE_MS = 1200
FLOOR_FRAMES = 40                          # 2 s of non-speech frames define the floor
FLOOR_PERCENTILE = 20
FLOOR_MULT_ON = 2.5                        # speech must be this many times the floor ...
FLOOR_MULT_OFF = 1.5                       # ... and silence is below this many times the floor
SILENCE_REL = 0.2                          # ... and below this share of the voice's own level
SPEECH_MIN_RMS = 60.0                      # absolute minimums, for a near-silent room
SILENCE_MIN_RMS = 35.0
# the transcription address: VA_WHISPER if set, else providers.transcription.url in the config file
WHISPER_MODEL = "whisper-gpu"
NO_SPEECH_PROB = 0.6                       # Whisper's own "this is not speech" estimate, per segment
MIC_SOURCE = os.environ.get("VA_MIC") or audio_devices.mic_source()  # echo-cancelled when the pair is loaded
RUNTIME = os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir())
PIDFILE = os.path.join(RUNTIME, "voice_overlay.pid")
LOGFILE = os.path.join(tempfile.gettempdir(), "voice_overlay.log")
LAST_QUESTION_WAV = os.path.join(tempfile.gettempdir(), "va_last_question.wav")

# what Whisper writes for silence and noise rather than for words
NOISE_TRANSCRIPTS = {"thank you", "thanks", "thank you very much", "obrigado", "obrigada", "ooh", "the",
                     "yeah", "bye", "ah", "pfft", "hmm", "um", "eh", "thanks for watching",
                     "thank you for watching", "thank you so much", "subtitles by the amara org"}
NOISE_PATTERN = re.compile(r"^(thank you|thanks)( for watching| so much| very much)?$|^(subtitles|legendas) ")


def frame_rms(pcm: np.ndarray) -> float:
    return math.sqrt(float(np.mean(pcm.astype(np.float64) ** 2))) if pcm.size else 0.0


def level_of(rms: float) -> float:
    """0..100 from an RMS value, mapped from -60 dBFS..-10 dBFS."""
    db = 20 * math.log10(max(rms, 1.0) / 32768)
    return max(0.0, min(100.0, (db + 60) / 50 * 100))


def is_noise_transcript(text: str) -> bool:
    """True when a transcript is not something the user said: empty, a Whisper hallucination, or
    any non-Latin letter (the card speaks Portuguese and English only; keyboard clicks come out as
    Cyrillic words)."""
    if any(c.isalpha() and "LATIN" not in unicodedata.name(c, "") for c in text):
        return True
    cleaned = re.sub(r"[^a-zá-ú ]", "", text.lower()).strip()
    return not cleaned or cleaned in NOISE_TRANSCRIPTS or bool(NOISE_PATTERN.match(cleaned))


class Endpointer:
    """Turn detection on 50 ms frames, with the noise floor taken from the room itself.

    update(rms) returns "onset" when speech starts, "end" when the turn is over, or "".
    The floor is the 20th percentile of the last 2 s of non-speech frames, so a loud room
    raises the thresholds with it instead of locking them above the voice.
    """

    def __init__(self, silence_ms: int):
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.floor_hist: collections.deque[float] = collections.deque(maxlen=FLOOR_FRAMES)
        self.speaking = False
        self.onset_run = 0
        self.silence_run = 0
        self.speech_frames = 0
        self.floor_on = 0.0
        self.speech_levels: list[float] = []
        self.warm_frames = 0

    def update(self, rms: float) -> str:
        if not self.speaking:
            self.floor_hist.append(rms)
            self.warm_frames += 1
            if self.warm_frames <= WARMUP_FRAMES:      # the shortcut's keypress click is not a question
                return ""
            floor = float(np.percentile(self.floor_hist, FLOOR_PERCENTILE))
            if rms > max(FLOOR_MULT_ON * floor, SPEECH_MIN_RMS):
                self.onset_run += 1
                if self.onset_run >= ONSET_FRAMES:
                    self.speaking = True
                    self.speech_frames = self.onset_run
                    self.floor_on = floor
                    self.silence_run = 0
                    return "onset"
            else:
                self.onset_run = 0
            return ""
        self.speech_frames += 1
        self.speech_levels.append(rms)
        level = float(np.median(self.speech_levels[-40:]))   # the voice's own level over the last 2 s
        if rms < max(FLOOR_MULT_OFF * self.floor_on, SILENCE_REL * level, SILENCE_MIN_RMS):
            self.silence_run += 1
            if self.silence_run >= self.silence_frames and self.speech_frames * FRAME_MS > MIN_SPEECH_MS:
                return "end"
        else:
            self.silence_run = 0
        return ""


def profile_silence_ms() -> int:
    """The silence window the panel's calibration measured between words, or the default."""
    try:
        with open(os.path.expanduser("~/.config/herdr/voice-profile.json")) as f:
            return int(json.load(f).get("silence_ms", DEFAULT_SILENCE_MS))
    except (OSError, ValueError):
        return DEFAULT_SILENCE_MS


async def open_mic() -> asyncio.subprocess.Process:
    """pw-record on the mic source, writing to stdout.

    With "-" as the output pw-record writes a Sun AU stream: a 24-byte header, then big-endian PCM16.
    The header is consumed here, and read_frame converts the samples to little-endian.
    """
    proc = await asyncio.create_subprocess_exec(
        "pw-record", "--target", MIC_SOURCE, "--rate", str(RATE), "--channels", "1", "--format", "s16", "-",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    head = await proc.stdout.readexactly(24)
    if head[:4] != b".snd":
        proc.terminate()
        raise RuntimeError(f"unexpected mic stream header {head[:4]!r}")
    return proc


def _be_to_le(raw: bytes) -> bytes:
    raw = raw[: len(raw) // 2 * 2]
    return np.frombuffer(raw, dtype=">i2").astype("<i2").tobytes()


async def read_frame(proc: asyncio.subprocess.Process) -> bytes:
    """One VAD frame of little-endian PCM16 from the mic."""
    return _be_to_le(await proc.stdout.readexactly(FRAME_BYTES))


ALLOWED_LANGUAGES = ("portuguese", "english")   # the only languages the card hears


def _whisper_once(wav: bytes, language: str) -> tuple[str, str, bool, float]:
    """One Whisper request. Returns (text, language, is non-speech, mean segment log-probability)."""
    boundary = "----VoiceOverlayBoundary"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
             f"Content-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n"]
    fields = [("model", WHISPER_MODEL), ("temperature", "0"), ("response_format", "verbose_json"),
              ("language", language)]
    for name, value in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(config.service_url("transcription", "VA_WHISPER"), data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode())
    segments = d.get("segments") or []
    no_speech = bool(segments) and min(float(s.get("no_speech_prob", 0.0)) for s in segments) > NO_SPEECH_PROB
    logprobs = [float(s["avg_logprob"]) for s in segments if "avg_logprob" in s]
    confidence = sum(logprobs) / len(logprobs) if logprobs else float("-inf")
    return d.get("text", "").strip(), d.get("language", ""), no_speech, confidence


def _whisper_sync(pcm_le: bytes) -> tuple[str, str]:
    """(text, language) for one utterance, from the dictation Whisper model.

    Whisper detects the language itself first. If it picks anything other than Portuguese or English
    (keyboard noise and short words such as "gerar" come out as French or Icelandic), the take is read
    again as Portuguese and as English, and the reading Whisper is more confident in is kept.
    Segments Whisper itself marks as non-speech are dropped.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm_le)
    wav = buf.getvalue()
    text, lang, no_speech, _ = _whisper_once(wav, "auto")
    if no_speech:
        return "", ""
    if lang in ALLOWED_LANGUAGES:
        return text, lang
    readings = [r for r in (_whisper_once(wav, code) for code in ("pt", "en")) if not r[2]]
    if not readings:
        return "", ""
    text, lang, _, _ = max(readings, key=lambda r: r[3])
    return text, lang


async def transcribe(pcm_le: bytes) -> tuple[str, str]:
    return await asyncio.to_thread(_whisper_sync, pcm_le)


class Bridge(QObject):
    """Signals from the audio/network thread to the Qt window."""
    level = Signal(float)
    line = Signal(str)        # the transcript of the turn
    question = Signal(str)    # the question shown in its bubble
    status = Signal(str)
    answer = Signal(str)      # answer text delta
    answer_reset = Signal()
    sources = Signal(list)    # [{"title", "link"}, ...] from a search
    finished = Signal()
    mood = Signal(str)        # idle | listen | think | speak, for the octopus
    agent = Signal(str)       # assistant | specialist: the colour and the fedora
    heard = Signal()          # the name was said: the right arm goes to the ear
    expand = Signal()        # a new card starts: the corner octopus grows back to the card's size
    setup_request = Signal(str)   # the settings panel asks for a name setup: "assistant" or "specialist"
    start_request = Signal()      # the play button: start a conversation and listen
    show_request = Signal()       # the application menu asks for the octopus to be shown


async def _record_and_transcribe(bridge: Bridge, stop: asyncio.Event,
                                 no_speech_ms: int = NO_SPEECH_MS, gate: bool = True) -> tuple[str, str, bool]:
    """Record one turn, end it on a pause, and transcribe it once with Whisper.

    Returns (text, language, heard). Only speech is kept: the pre-roll before the onset is included, the
    silence before it is not, and nothing is sent to Whisper if no speech started.
    """
    mic = await open_mic()

    async def stopper():
        await stop.wait()
        mic.terminate()

    stop_task = asyncio.create_task(stopper())
    bridge.status.emit("A ouvir…")
    turn = bytearray()
    pre_roll: collections.deque[bytes] = collections.deque(maxlen=PRE_ROLL_FRAMES)
    ep = Endpointer(profile_silence_ms())
    frames_seen = 0
    lead_task: asyncio.Task | None = None   # the early name check, running on the opening of the speech
    lead_next = LEAD_CHECK_FRAMES           # the speech length at which the next early check starts (0 when done)
    heard_name = False                      # the name was heard early: the arm is already at the ear
    try:
        await read_frame(mic)  # the first frame is a device-open click
        while True:
            try:
                chunk = await read_frame(mic)
            except asyncio.IncompleteReadError as e:
                chunk = _be_to_le(e.partial)
                if chunk and ep.speaking:
                    turn += chunk
                break
            rms = frame_rms(np.frombuffer(chunk, dtype=np.int16))
            bridge.level.emit(level_of(rms))
            frames_seen += 1
            if not ep.speaking:
                pre_roll.append(chunk)
                if ep.update(rms) == "onset":
                    print(f"onset: rms {rms:.0f}, floor {ep.floor_on:.0f}", flush=True)
                    turn += b"".join(pre_roll)
                    pre_roll.clear()
                    bridge.status.emit("A ouvir… (a falar)")
                elif frames_seen * FRAME_MS >= no_speech_ms:
                    break
                continue
            turn += chunk
            if gate and not heard_name and lead_task is None and lead_next and ep.speech_frames >= lead_next:
                lead_task = asyncio.create_task(transcribe(bytes(turn)))   # the opening only, in the background
                lead_next = LEAD_CHECK2_FRAMES if lead_next == LEAD_CHECK_FRAMES else 0
            if lead_task is not None and lead_task.done():
                lead = lead_task.result()[0]
                lead_task = None
                if config.addressee(lead, config.load()) is not None or is_stop_request(lead):
                    heard_name = True
                    print(f"heard early ({ep.speech_frames * FRAME_MS} ms): {lead!r}", flush=True)
                    bridge.heard.emit()                 # the name is heard: the arm goes to the ear now
                else:
                    # a short lead is often too little for Whisper (1.3 s gave "Z, K, R"): this never ends the turn
                    print(f"early check, no name yet ({ep.speech_frames * FRAME_MS} ms): {lead!r}", flush=True)
            if ep.update(rms) == "end" or ep.speech_frames * FRAME_MS >= MAX_TURN_MS:
                print(f"turn end: speech {ep.speech_frames * FRAME_MS} ms, floor {ep.floor_on:.0f}, "
                      f"silence after {ep.silence_run * FRAME_MS} ms", flush=True)
                break
    finally:
        stop_task.cancel()
        if mic.returncode is None:
            mic.terminate()
    await mic.wait()

    if not ep.speaking:
        return "", "", False
    write_wav(LAST_QUESTION_WAV, np.frombuffer(bytes(turn), dtype=np.int16), RATE)
    bridge.status.emit("A transcrever…")
    text, lang = await transcribe(bytes(turn))
    if text:
        bridge.line.emit(text)
    return text, lang, True


LEAD_FRAMES = 24          # 1.2 s of new sound decides whether it is the room echoing the answer or the user
LEAD_CHECK_FRAMES = 26    # 1.3 s of speech: the first early name check, while the user is still talking
LEAD_CHECK2_FRAMES = 52   # 2.6 s: Whisper needs about this much to write a whole name, so a second check


def _is_echo(lead: str, spoken: str) -> bool:
    """True when what was heard is not the user talking over the answer: the agent's own voice coming
    back through the room, or noise (Whisper writes 'thank you' for playback it cannot transcribe)."""
    if is_noise_transcript(lead):
        return True
    words = re.findall(r"[a-zà-ú]+", lead.lower())
    if not words:
        return True
    said = set(re.findall(r"[a-zà-ú]+", spoken.lower()))
    return sum(1 for w in words if w in said) / len(words) >= 0.5


def residual_words(heard: str, spoken: str) -> str:
    """What was heard without the words of the spoken answer: the user's own words inside the room's echo."""
    said = set(re.findall(r"[a-zà-ú]+", spoken.lower()))
    return " ".join(w for w in re.findall(r"[a-zà-ú]+", heard.lower()) if w not in said)


def _addresses_assistant(text: str) -> bool:
    """A turn counts only when it starts with the assistant's name. With no name set, nothing said over an answer
    is taken, so the echo of the answer never becomes a question."""
    cfg = config.load()
    return bool(cfg["assistant"].get("name")) and config.addressed_to(text, cfg)


def _barge_in(text: str) -> bool:
    """Talking over the answer stops the voice only for the name or a stop request."""
    return _addresses_assistant(text) or is_stop_request(text)


def _strip_name(text: str) -> str:
    parts = text.split(None, 1)
    return parts[1] if len(parts) > 1 else text


def _question_of(text: str) -> str:
    """The question without the name it starts with (when a name is set)."""
    return _strip_name(text) if config.load()["assistant"].get("name") else text


async def stop_listener(done: asyncio.Event, cancel: asyncio.Event, closing: asyncio.Event,
                        spoken: dict, pending: list[str]) -> None:
    """Listen until `done` (the answer turn is over).

    - A stop request closes the card (`closing`) and stops the playback.
    - Speech that starts while the answer plays is judged on its first 1.2 s: if those words are the
      answer being spoken, it is echo and ignored; otherwise the user is talking over the agent, the
      playback stops (`cancel`), and what they said is queued as the next question (`pending`).
    """
    mic = await open_mic()
    try:
        await read_frame(mic)
        ep = Endpointer(profile_silence_ms())
        pre: collections.deque[bytes] = collections.deque(maxlen=PRE_ROLL_FRAMES)
        utter: list[bytes] = []
        decided = echo = False

        async def finish_utterance(echo_only: bool = False):
            if not decided:
                return
            text, _ = await transcribe(b"".join(utter))
            print(f"listener heard: {text!r} echo={echo_only}", flush=True)
            if echo_only:
                # the answer's own voice came back through the room; a stop word spoken inside it still counts
                if is_stop_request(residual_words(text, spoken.get("text", ""))):
                    closing.set()
                    cancel.set()
                return
            if is_stop_request(text):
                closing.set()
                cancel.set()
            elif text and not is_noise_transcript(text) and _addresses_assistant(text):
                pending.append(_strip_name(text))

        while not done.is_set():
            chunk = await read_frame(mic)
            rms = frame_rms(np.frombuffer(chunk, dtype=np.int16))
            if not ep.speaking:
                pre.append(chunk)
                if ep.update(rms) == "onset":
                    utter = list(pre)
                    pre.clear()
                    decided = echo = False
                continue
            utter.append(chunk)
            if spoken.get("active") and not decided and len(utter) >= LEAD_FRAMES:
                decided = True
                lead, _ = await transcribe(b"".join(utter[:LEAD_FRAMES]))
                echo = _is_echo(lead, spoken.get("text", ""))
                print(f"barge-in lead: {lead!r} echo={echo} spoken={spoken.get('text', '')[:120]!r}", flush=True)
                if not echo and _barge_in(lead):
                    cancel.set()             # the user addresses the assistant over the answer: stop the voice
            if ep.update(rms) == "end" or len(utter) >= MAX_TURN_MS // FRAME_MS:
                await finish_utterance(echo_only=decided and echo)
                ep = Endpointer(profile_silence_ms())
                utter, pre, decided, echo = [], collections.deque(maxlen=PRE_ROLL_FRAMES), False, False
        if ep.speaking and utter:             # the answer ended mid-sentence: keep what was said
            await finish_utterance(echo_only=decided and echo)
    finally:
        if mic.returncode is None:
            mic.terminate()
        await mic.wait()


LANGUAGE_NOTE = " (Responde só em português europeu ou em inglês.)"   # added to the question on a retry
DIAGRAM_SAID = "Abri o diagrama no navegador."   # what the card and the voice say when a diagram file was opened


def _answer_in_language(text: str) -> bool:
    """True when the answer is English or European Portuguese. Too short to tell: it passes."""
    if len(text.strip()) < 30:
        return True
    return speech.guess_language(text) in ("pt", "en")


async def _answer_turn(bridge: Bridge, question: str, history: list[tuple[str, str]],
                       closing: asyncio.Event, lang: str | None, route: SpecialistSession | None,
                       stop: asyncio.Event) -> tuple[str, list[str]]:
    """Answer one question and speak it. Returns (answer text, utterances said over the answer).
    `stop` is the card closing (shortcut or Esc): it cancels a playing answer at once."""
    done = asyncio.Event()     # set when the answer turn is over
    cancel = asyncio.Event()   # set when the playback must stop: a stop request or the user talking over it
    spoken = {"active": False, "text": ""}
    pending: list[str] = []
    listener = None
    closer = None
    bridge.answer_reset.emit()
    bridge.status.emit("A responder…")
    bridge.agent.emit("specialist" if route is not None else "assistant")
    bridge.mood.emit("think")

    async def speak(t: str) -> None:
        spoken["text"], spoken["active"] = t, True
        bridge.mood.emit("speak")
        try:
            await speech.speak(t, cancel=cancel)
        finally:
            spoken["active"] = False
            bridge.mood.emit("think")

    try:
        listener = asyncio.create_task(stop_listener(done, cancel, closing, spoken, pending))

        async def close_when_stopped():
            await stop.wait()                 # shortcut or Esc: the card closes and the answer stops now
            closing.set()
            cancel.set()

        closer = asyncio.create_task(close_when_stopped())

        async def produce(q: str) -> str:
            if needs_screen(q):
                # a question about a monitor: the screenshot goes to the model with the question
                bridge.status.emit("A ver o ecrã…")
                out = await asyncio.to_thread(ask_screen, q)
                bridge.answer.emit(out)
                return out
            if route is not None:
                # the specialist answers; the octopus types on the laptop while it works
                bridge.mood.emit("write")
                out = await specialist_answer(q, route, bridge.status.emit, speak)
                bridge.answer.emit(out)
                return out
            return await answer(q,
                                on_text=bridge.answer.emit,
                                on_status=bridge.status.emit,
                                on_sources=bridge.sources.emit,
                                on_announce=speak,
                                history=history[-HISTORY_TURNS:], lang=lang, on_phase=bridge.mood.emit)

        text = await produce(question)
        if text.strip() and not _answer_in_language(text):
            # the answer came out in another language: ask once more with the rule stated, then keep it on the card
            print(f"answer not in pt/en, asking again: {text[:60]!r}", flush=True)
            bridge.answer_reset.emit()
            bridge.status.emit("A responder outra vez…")
            text = await produce(question + LANGUAGE_NOTE)
            if not _answer_in_language(text):
                print(f"answer still not in pt/en, not spoken: {text[:60]!r}", flush=True)
                bridge.status.emit("A resposta não ficou em português nem em inglês; não a leio.")
                return text, pending
        if text.strip() and open_tool.diagram_files(text):
            # a diagram file: the card and the voice say it is open; the file name is kept for the session to open
            bridge.answer_reset.emit()
            bridge.answer.emit(DIAGRAM_SAID)
            if not cancel.is_set():
                await speak(DIAGRAM_SAID)
        elif text.strip() and not cancel.is_set():
            await speak(text)
        bridge.status.emit("")
        return text, pending
    finally:
        done.set()
        if closer is not None:
            closer.cancel()
        if listener is not None:
            try:
                await asyncio.wait_for(listener, timeout=8)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                listener.cancel()


YES_WORDS = {"sim", "yes", "pode", "podes", "claro", "ok", "okay", "isso", "pode ser"}
FOLDER_WORDS_RE = re.compile(r"\b(pasta|pastas|folder|folders|ficheiro|ficheiros|file|files|diret[oó]rio|directory)\b",
                              re.IGNORECASE)


async def _say(bridge: Bridge, text: str) -> None:
    """The card shows a line and the voice says it."""
    bridge.answer_reset.emit()
    bridge.answer.emit(text)
    await speech.speak(text)


async def ask_yes(bridge: Bridge, stop: asyncio.Event, prompt: str, tries: int = 2) -> bool:
    """Say a question and listen for a yes. Silence or noise asks again; anything else that is not a yes is a no."""
    for _ in range(tries):
        await _say(bridge, prompt)
        said, _, heard = await _record_and_transcribe(bridge, stop, FOLLOWUP_MS, gate=False)
        if not heard or is_noise_transcript(said):
            continue
        return config.normalize(said) in YES_WORDS or config.first_word(said) in YES_WORDS
    return False


async def confirm_pane_send(bridge: Bridge, stop: asyncio.Event) -> None:
    """A prompt the assistant staged for a herdr pane is read back, and sent only after a spoken yes."""
    staged = herdr_pane.pending()
    if staged is None:
        return
    if await ask_yes(bridge, stop, herdr_pane.readback(staged)):
        await asyncio.to_thread(herdr_pane.confirm)
        await _say(bridge, f"Enviado para {staged['title']}.")
    else:
        herdr_pane.cancel()
        await _say(bridge, "Não enviei.")


async def folder_gate(bridge: Bridge, question: str, stop: asyncio.Event) -> bool:
    """A request about a folder reaches the specialist only if the folder is approved. A new folder needs a yes,
    said by voice to the front desk, and then it is added to the config for good."""
    if not FOLDER_WORDS_RE.search(question):
        return True                             # no folder is mentioned: nothing to approve
    name = await specialist_mod.folder_named(question)
    if not name:
        return True
    matches = await asyncio.to_thread(specialist_mod.find_folders, name)
    if not matches:
        await _say(bridge, f"Não encontrei nenhuma pasta chamada {name}.")
        return False
    return await approve_path(bridge, matches[0], stop)


async def approve_path(bridge: Bridge, path: str, stop: asyncio.Event) -> bool:
    """A folder the specialist may read and write: approved already, or by a spoken yes (then saved to the config)."""
    cfg = config.load()
    approved = {os.path.realpath(os.path.expanduser(f)) for f in cfg["access"]["folders"]}
    if os.path.realpath(path) in approved:
        return True
    for _ in range(2):
        await _say(bridge, f"Posso ler a pasta {path}? Só leio; não escrevo nela. Diz sim ou não.")
        said, _, heard = await _record_and_transcribe(bridge, stop, FOLLOWUP_MS, gate=False)
        if not heard or is_noise_transcript(said):
            continue
        first = config.first_word(said)
        if first in YES_WORDS or config.normalize(said) in YES_WORDS:
            cfg["access"]["folders"].append(path)
            config.save(cfg)
            return True
        break
    await _say(bridge, "Ok, não uso essa pasta.")
    return False


SETUP_SENTENCES = [
    "{name}, que horas são em Lisboa?",
    "{name}, qual é a capital de Portugal?",
    "{name}, o que está no meu ecrã?",
    "{name}, pesquisa o tempo em Lisboa.",
    "{name}, para.",
    "{name}, quanto é dois mais dois?",
    "{name}, lê-me as notícias de hoje.",
    "{name}, obrigado.",
]
SPECIALIST_SENTENCES = [
    "{name}, desenha um diagrama do fluxo de pedidos.",
    "{name}, que ficheiros há na pasta de exemplo?",
    "{name}, explica em detalhe como funciona o router.",
    "{name}, abre a pasta Outbox.",
    "{name}, resume o texto do ficheiro TODO.",
    "{name}, mostra o estado do repositório.",
    "{name}, lista os ficheiros da pasta de trabalho.",
    "{name}, obrigado.",
]


async def setup_main(bridge: Bridge, stop: asyncio.Event, target: str = "assistant") -> None:
    """Name setup for the assistant ("assistant") or the specialist ("specialist"). Each sentence is shown in the box
    as a question, the mic listens while the bars show, and what Whisper wrote is shown after each take. A take
    counts when it has at least two words; the name forms are the first words that at least two takes agree on
    (config.name_forms_from_takes), so a name the mic always writes one way ("Todd" for Thoth) is learnt too. The
    forms are saved in the config, and setup_done is set. Esc or the shortcut stops it."""
    cfg = config.load()
    section = cfg["assistant"] if target == "assistant" else cfg["providers"]["specialist"]
    name = section["name"]
    sentences = SETUP_SENTENCES if target == "assistant" else SPECIALIST_SENTENCES
    takes: list[str] = []
    try:
        bridge.question.emit(f"Configurar {name}: vou pedir {len(sentences)} frases que começam pelo nome.")
        bridge.answer_reset.emit()
        bridge.answer.emit("Quando a bolinha começar a gravar, diz a frase que aparece em cima.")
        for number, template in enumerate(sentences, 1):
            sentence = template.format(name=name)
            while True:
                if stop.is_set():
                    return
                bridge.question.emit(f"Frase {number} de {len(sentences)}: {sentence}")
                bridge.answer_reset.emit()
                bridge.answer.emit("<i>a gravar…</i>")
                said, _, was_heard = await _record_and_transcribe(bridge, stop, NO_SPEECH_MS, gate=False)
                print(f"setup {target} heard: {said!r} (sentence {number})", flush=True)
                if not was_heard:
                    bridge.answer_reset.emit()
                    bridge.answer.emit("Não ouvi nada. Verifica o microfone (mudo?) e diz a frase outra vez.")
                    continue
                if is_noise_transcript(said) or len(config.normalize(said).split()) < 2:
                    continue                   # room noise or a single word: listen again
                takes.append(said)
                bridge.answer_reset.emit()
                bridge.answer.emit(f"Ouvi: <b>{html.escape(said)}</b> ✓")
                await asyncio.sleep(1.5)
                break
        forms = config.name_forms_from_takes(takes)
        section["wake_variants"] = forms
        section["setup_done"] = bool(forms)
        config.save(cfg)
        bridge.answer_reset.emit()
        if forms:
            bridge.answer.emit("Ok, já tenho um nome.")
        else:
            bridge.answer.emit("Não ouvi o nome em frases suficientes. Corre a configuração outra vez.")
    finally:
        bridge.mood.emit("idle")
        bridge.finished.emit()


async def session_main(bridge: Bridge, stop: asyncio.Event) -> None:
    """A conversation: each turn is heard, answered and spoken; the card stays open for the next
    turn and closes after FOLLOWUP_MS of silence, on a stop request, on Esc, or the shortcut again.
    A question said over an answer is taken as the next turn without waiting."""
    history: list[tuple[str, str]] = []
    closing = asyncio.Event()   # set when a stop request closes the conversation
    queued = ""                 # a question said over the previous answer
    agent_qwen = False          # the agent that answered last: the specialist stays on until the assistant is asked back
    spec = SpecialistSession()  # one per card: its opencode session lives in the specialist folder
    try:
        while True:
            lang = None             # a question said over an answer: its language is not known yet
            fresh = False           # set when this turn was heard now (not queued): its name is taken off
            if queued:
                text, queued = queued, ""
            else:
                no_speech = FOLLOWUP_MS if history else NO_SPEECH_MS
                text = ""
                fresh = True
                bridge.mood.emit("listen")
                cfg = config.load()
                name = cfg["assistant"]["name"]
                noise = 0
                while True:
                    text, lang, heard = await _record_and_transcribe(bridge, stop, no_speech)
                    print(f"heard: {text!r} lang={lang}", flush=True)
                    if not heard:                   # silence: nothing was said, so the conversation ends
                        text = ""
                        break
                    if not text or is_noise_transcript(text):
                        noise += 1
                        if noise >= HEARD_ATTEMPTS:
                            text = ""
                            break
                        bridge.status.emit("Só ruído — a ouvir de novo…")
                        continue
                    if config.addressee(text, cfg) is None and not is_stop_request(text):   # stop always works
                        print(f"not addressed: {text!r}", flush=True)
                        bridge.status.emit(f"Começa a pergunta por {name} ou {cfg['providers']['specialist']['name']}.")
                        continue
                    break
                if not text or is_noise_transcript(text):
                    bridge.status.emit("Até à próxima." if history else "Não ouvi nada.")
                    return
            bare_stop = fresh and config.normalize(_question_of(text)) == "para"   # "Zeta, para." closes the card
            if is_stop_request(text) or bare_stop:   # "stop", "goodbye", "that's all": close the card
                bridge.status.emit("Até à próxima.")
                return
            who = config.addressee(text, cfg) if fresh else None   # read before the name is taken off
            if fresh:
                text = _question_of(text)     # the name is said, not asked: the question is the rest
                bridge.heard.emit()           # the arm goes to the ear: what is said now goes to the agent
            bridge.question.emit(text)        # the card always shows what was heard, before any answer
            if who == "specialist":
                use_qwen = True               # the specialist's name always goes to the specialist
            elif who == "assistant":
                use_qwen = False              # the assistant's name always goes to the front desk, sticky or not
            else:
                use_qwen = not needs_screen(text) and await specialist_decide(text, agent_qwen)
            if fresh and open_tool.open_name(text):
                wanted = open_tool.open_name(text)
                matches = await asyncio.to_thread(open_tool.approved_folders_named, wanted)
                if not matches:
                    found = await asyncio.to_thread(specialist_mod.find_folders, wanted)
                    if found and await approve_path(bridge, found[0], stop):
                        matches = [found[0]]
                app = None if matches else await asyncio.to_thread(open_tool.find_app, wanted)
                if matches and open_tool.open_path(matches[0]):
                    await _say(bridge, f"Abri {matches[0]}.")
                elif app is not None:
                    if await ask_yes(bridge, stop, f"Abro {app['name']}? Diz sim ou não."):
                        open_tool.launch_app(app["id"])
                        await _say(bridge, f"Abri {app['name']}.")
                    else:
                        await _say(bridge, "Não abri.")
                else:
                    await _say(bridge, f"Não encontrei {wanted}.")
                bridge.status.emit("")
                continue
            if use_qwen and not await folder_gate(bridge, text, stop):
                bridge.status.emit("")
                continue
            answer_text, said = await _answer_turn(bridge, text, history, closing, lang,
                                                   spec if use_qwen else None, stop)
            await confirm_pane_send(bridge, stop)
            for diagram in open_tool.diagram_files(answer_text):
                open_tool.open_path(diagram)
            agent_qwen = use_qwen          # sticky: the next question stays with the same agent
            if closing.is_set():
                bridge.status.emit("Até à próxima.")
                return
            history.append((text, answer_text))
            queued = said[0] if said else ""
    except Exception as e:  # show the failure in the card instead of dying silently
        bridge.status.emit(f"Erro: {e or type(e).__name__ + ' (serviço de voz sem resposta)'}")
    finally:
        bridge.mood.emit("idle")
        bridge.finished.emit()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _running_pid() -> int:
    """The pid of a running overlay, or 0."""
    try:
        pid = int(open(PIDFILE).read().strip())
    except (FileNotFoundError, ValueError):
        return 0
    return pid if pid_alive(pid) else 0


def _spawn() -> None:
    """Start the overlay as its own process, detached from the caller."""
    with open(LOGFILE, "a") as log:
        # Wayland clients can't place their own windows; under XWayland Qt can, so the
        # card lands on the primary screen instead of wherever KWin puts it
        env = {**os.environ, "QT_QPA_PLATFORM": "xcb"}
        subprocess.Popen([sys.executable, "-m", "voiceassistant"], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env)


def toggle() -> None:
    """Called by the keybind: stop a running card, or start one."""
    pid = _running_pid()
    if pid:
        os.kill(pid, signal.SIGUSR1)
        return
    _spawn()


def show() -> None:
    """Called by the application menu entry: show the octopus again, or start it if it is not running."""
    pid = _running_pid()
    if pid:
        os.kill(pid, signal.SIGUSR2)
        return
    _spawn()


def run_gui(setup: bool = False, target: str = "assistant") -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)    # hidden to the tray is not the end of the app
    bridge = Bridge()
    win = Overlay(bridge, on_escape=app.quit)
    win.place()
    win.show()
    restart = {"requested": False}

    def on_restart() -> None:
        restart["requested"] = True
        app.quit()

    tray = make_tray(win, on_restart, app.quit)     # the tray icon: left click shows or hides, right click menu
    bridge.expand.connect(win.expand)
    bridge.show_request.connect(win.show_window)

    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))

    loop = asyncio.new_event_loop()
    current = {"stop": None, "active": False}   # the card open now: its stop event, and whether it is running
    start_card = threading.Event()

    def on_signal(*_):
        # the shortcut: stops the open card, or opens a new one when only the corner octopus is left
        if current["active"] and current["stop"] is not None:
            loop.call_soon_threadsafe(current["stop"].set)
        else:
            start_card.set()

    signal.signal(signal.SIGUSR1, on_signal)
    # the application menu's entry: the octopus is shown again (see show())
    signal.signal(signal.SIGUSR2, lambda *_: bridge.show_request.emit())
    # Python runs signal handlers only between bytecodes; keep the interpreter ticking under Qt's loop
    pump = QTimer()
    pump.timeout.connect(lambda: None)
    pump.start(200)

    setup_name: dict = {"target": None}   # a name setup asked for from the settings panel, run as the next card

    def on_setup(name: str) -> None:
        # from the settings panel: an open card stops, and the name setup runs as the next card
        setup_name["target"] = name
        if current["active"] and current["stop"] is not None:
            loop.call_soon_threadsafe(current["stop"].set)
        start_card.set()

    bridge.setup_request.connect(on_setup)

    def on_start() -> None:
        # the play button: a card opens and listens; while a card is open this does nothing
        if not current["active"]:
            start_card.set()

    bridge.start_request.connect(on_start)

    async def one_card() -> None:
        if setup:
            await setup_main(bridge, current["stop"], target)
        elif setup_name["target"]:
            name, setup_name["target"] = setup_name["target"], None
            await setup_main(bridge, current["stop"], name)
        else:
            await session_main(bridge, current["stop"])

    def worker_main() -> None:
        # a card per shortcut press; the setup runs once and closes the app
        while True:
            start_card.wait()
            start_card.clear()
            current["stop"] = asyncio.Event()     # exists before the card counts as active, so a press always finds it
            current["active"] = True
            bridge.expand.emit()
            try:
                loop.run_until_complete(one_card())
            finally:
                current["active"] = False
            if setup:
                break

    worker = threading.Thread(target=worker_main, daemon=True)
    start_card.set()
    worker.start()

    def finish():
        if setup:
            win.fade_out(FADE_MS, app.quit)
        else:
            win.to_corner()
    bridge.finished.connect(lambda: QTimer.singleShot(FADE_DELAY_MS, finish))

    app.exec()
    tray.hide()
    if current["stop"] is not None:
        loop.call_soon_threadsafe(current["stop"].set)
    if current["active"]:
        worker.join(timeout=3)
    if restart["requested"]:
        # the same process id is kept, so the pid file stays valid for the new start
        os.execv(sys.executable, [sys.executable, "-m", "voiceassistant", *sys.argv[1:]])
    try:
        os.remove(PIDFILE)
    except FileNotFoundError:
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--toggle", action="store_true", help="start the card, or stop a running one")
    p.add_argument("--show", action="store_true", help="show the octopus (start it if it is not running)")
    p.add_argument("--setup", action="store_true", help="run the assistant's name setup again")
    p.add_argument("--setup-specialist", action="store_true", help="run the specialist's name setup")
    args = p.parse_args()
    config.ensure_file()                  # first run: the shipped example becomes the user's config
    if args.show:
        show()
    elif args.toggle:
        toggle()
    elif args.setup_specialist:
        run_gui(setup=True, target="specialist")
    else:
        run_gui(setup=args.setup or not config.load()["assistant"]["setup_done"])


if __name__ == "__main__":
    main()
