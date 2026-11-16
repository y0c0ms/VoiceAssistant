"""stop.py — the spoken stop command.

Only the word "stop" stops the playback or closes the card. Nothing else does: Portuguese "para" and
phrases like "that's all" or "goodbye" do not, because they are common words in ordinary speech.
"""
import re

_STOP_RE = re.compile(r"\bstop\b", re.IGNORECASE)


def is_stop_request(text: str) -> bool:
    return bool(_STOP_RE.search(text))
