#!/usr/bin/env python3
"""voice_chat.py — live voice conversation with LFM3-Omni-2B over llama-server /v1/realtime.

Protocol per llama.cpp tools/server/README.md (Realtime conversation):
session.update -> input_audio_buffer.append* -> input_audio_buffer.commit
-> response.create -> response.output_text.delta / response.output_audio.delta
(base64 PCM16 mono @ audio.output.format.rate) -> response.done.

llama-swap returns 404 for /v1/realtime, so the client loads the model through
llama-swap and then connects to the lfm3 llama-server port it spawned.

Usage:
  python3 voice_chat.py                  # record mic until Enter, answer by voice
  python3 voice_chat.py --text "hi"      # text question, answered by voice
  python3 voice_chat.py --wav q.wav      # PCM16 wav question
  python3 voice_chat.py --no-play        # save answer.wav only
"""
import argparse
import asyncio
import base64
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.request
import wave

import numpy as np
import websockets

SWAP = os.environ.get("VA_SWAP", "http://127.0.0.1:11435")
MODEL_PATH_MARK = "LFM3-Omni-2B-Q8_0.gguf"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_realtime_url() -> str:
    """Load lfm3-omni through llama-swap, then find the llama-server --port it spawned.

    Override with VA_WS=ws://host:port/v1/realtime.
    """
    if os.environ.get("VA_WS"):
        return os.environ["VA_WS"]
    req = urllib.request.Request(
        f"{SWAP}/v1/chat/completions",
        data=json.dumps({"model": "lfm3-omni",
                         "messages": [{"role": "user", "content": "ok"}],
                         "max_tokens": 1}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=600).read()
    for line in subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines():
        if "llama-server" in line and MODEL_PATH_MARK in line:
            toks = line.split()
            return f"ws://127.0.0.1:{toks[toks.index('--port') + 1]}/v1/realtime"
    raise SystemExit("lfm3-omni llama-server not found after load")


def record_mic(path: str, rate: int) -> None:
    """Record the default mic to a wav until Enter is pressed."""
    src = subprocess.run(["pactl", "get-default-source"], capture_output=True, text=True).stdout.strip() or "@DEFAULT_SOURCE@"
    print(f"🎙  recording from {src} — press Enter to stop", flush=True)
    proc = subprocess.Popen(
        ["pw-record", "--target", src, "--rate", str(rate), "--channels", "1", "--format", "s16", path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        input()
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)


def read_wav(path: str, rate: int) -> np.ndarray:
    with wave.open(path, "rb") as f:
        assert f.getsampwidth() == 2, "need PCM16 wav"
        x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
        x = x.reshape(-1, f.getnchannels()).mean(axis=1)
        sr = f.getframerate()
    if sr != rate:
        n = int(round(len(x) * rate / sr))
        x = np.interp(np.arange(n) * sr / rate, np.arange(len(x)), x)
    return np.clip(x, -32768, 32767).astype(np.int16)


def write_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


async def run_turn(args) -> None:
    url = resolve_realtime_url()
    async with websockets.connect(url, max_size=None) as ws:
        first = json.loads(await ws.recv())
        if first.get("type") == "error":
            raise SystemExit(f"server error: {first['error']['message']}")
        sess = first["session"]
        in_rate = sess["audio"]["input"]["format"]["rate"]
        out_rate = sess["audio"].get("output", {}).get("format", {}).get("rate", 24000)
        print(f"connected {url} in={in_rate}Hz out={out_rate}Hz", flush=True)

        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime",
            "instructions": args.instructions,
            "output_modalities": args.modalities.split(","),
        }}))
        ev = json.loads(await ws.recv())
        if ev["type"] == "error":
            raise SystemExit(f"session.update error: {ev['error']['message']}")

        if args.text:
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": args.text}]}}))
        else:
            wav = args.wav
            if wav is None:
                wav = os.path.join(tempfile.gettempdir(), "va_question.wav")
                record_mic(wav, in_rate)
            pcm = read_wav(wav, in_rate)
            step = in_rate // 10  # 100 ms chunks, as from a mic
            for j in range(0, len(pcm), step):
                chunk = pcm[j:j + step]
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(chunk.tobytes()).decode()}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))

        audio, t_req, t_first_audio = [], time.perf_counter(), None
        async for raw in ws:
            ev = json.loads(raw)
            t = ev["type"]
            if t == "response.output_text.delta":
                print(ev["delta"], end="", flush=True)
            elif t == "response.output_audio.delta":
                t_first_audio = t_first_audio or time.perf_counter()
                audio.append(np.frombuffer(base64.b64decode(ev["delta"]), dtype=np.int16))
            elif t == "conversation.item.input_audio_transcription.completed":
                print(f"[heard] {ev.get('transcript', '')}", flush=True)
            elif t == "error":
                raise SystemExit(f"\nserver error: {ev['error']['message']}")
            elif t == "response.done":
                break
        print()
        if t_first_audio:
            print(f"first audio after {1000 * (t_first_audio - t_req):.0f} ms", flush=True)
        if audio:
            pcm_out = np.concatenate(audio)
            out_path = os.path.join(OUT_DIR, "answer.wav")
            write_wav(out_path, pcm_out, out_rate)
            print(f"audio {len(pcm_out) / out_rate:.2f}s -> {out_path}", flush=True)
            if not args.no_play:
                subprocess.run(["paplay", out_path])
        else:
            print("(no audio in response)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wav", help="PCM16 wav question instead of the mic")
    p.add_argument("--text", help="text question instead of voice")
    p.add_argument("--instructions", default=(
        "Respond in text and audio. Always answer in the language the user speaks. "
        "If the user speaks European Portuguese (pt-PT), answer in European Portuguese, "
        "not Brazilian Portuguese and not English."),
                   help="system prompt; the default makes the answer follow the user's language")
    p.add_argument("--modalities", default="text,audio", help="text | audio | text,audio")
    p.add_argument("--no-play", action="store_true")
    asyncio.run(run_turn(p.parse_args()))


if __name__ == "__main__":
    main()
