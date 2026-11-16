"""audio_devices.py: which PipeWire devices the voice card uses.

The echo-cancelled pair (aec_tts_src / aec_tts_sink) lets the card hear the user over its own speech through
the speakers: TTS plays into aec_tts_sink, whose echo reference the mic source subtracts. If the pair is not
loaded (it is loaded at runtime, so it does not survive a reboot), the card falls back to the noise-suppressed
herdr source and the default output, which means headphones are needed.
"""
import functools
import subprocess

AEC_SOURCE = "aec_tts_src"
AEC_SINK = "aec_tts_sink"
FALLBACK_SOURCE = "herdr_clean_source"


def pick_device(preferred: str, fallback: str | None, available: set[str]) -> str | None:
    """`preferred` when it exists, else `fallback` (None means the system default)."""
    if preferred in available:
        return preferred
    return fallback


@functools.lru_cache(maxsize=1)
def _available() -> frozenset[str]:
    out = subprocess.run(["pactl", "list", "short", "sinks"], capture_output=True, text=True).stdout
    out += subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) > 1:
            names.add(parts[1])
    return frozenset(names)


def mic_source() -> str:
    return pick_device(AEC_SOURCE, FALLBACK_SOURCE, set(_available())) or "@DEFAULT_SOURCE@"


def playback_sink() -> str | None:
    """None means the default output (paplay's own default)."""
    return pick_device(AEC_SINK, None, set(_available()))
