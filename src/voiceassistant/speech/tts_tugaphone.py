#!/usr/bin/env python3
"""tts_tugaphone.py — European Portuguese speech with the Phoonnx dii tugaphone voice
(Piper/VITS, local ONNX, CC BY-NC-ND 4.0).

Runs in the phoonnx venv:
  .venv-phoonnx/bin/python tts_tugaphone.py "Lisboa tem quinhentos e quarenta e um mil habitantes." -o out.wav
"""
import argparse
import os
import wave

from phoonnx.voice import TTSVoice

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "voices", "tugaphone"))   # repo root / voices
# the European Portuguese voices on disk, by name (the settings pick one)
VOICES = {
    "dii": (os.path.join(VOICE_DIR, "dii_pt-PT.onnx"), os.path.join(VOICE_DIR, "dii_pt-PT.json")),
    "miro": (os.path.join(VOICE_DIR, "miro_pt-PT.onnx"), os.path.join(VOICE_DIR, "miro_pt-PT.json")),
}


def synth(text: str, out_path: str, voice: str = "dii") -> None:
    model, config = VOICES[voice]
    tts = TTSVoice.load(model_path=model, config_path=config)
    with wave.open(out_path, "wb") as wf:
        tts.synthesize_wav(text, wf)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("-o", "--out", default="tugaphone_dii.wav")
    p.add_argument("--voice", default="dii", choices=sorted(VOICES))
    a = p.parse_args()
    synth(a.text.strip(), a.out, a.voice)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
