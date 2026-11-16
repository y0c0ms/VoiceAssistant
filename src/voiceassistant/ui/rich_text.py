"""rich_text.py — markdown and LaTeX for the overlay, and plain text for the voice.

render_html(md) -> HTML for a Qt label: markdown through Qt's own converter, and each LaTeX
formula ($...$ inline, $$...$$ display) as a PNG from matplotlib's mathtext, embedded inline.
plain(md) -> the same text without markup, for speech.
"""
import base64
import html as html_lib
import io
import re

# matplotlib is imported when the first formula is rendered, not at startup
from PySide6.QtGui import QTextDocument

_LATEX = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$", re.DOTALL)
_PNG_CACHE: dict[tuple[str, bool], str] = {}


def _formula_png(expr: str, display: bool) -> str:
    """Base64 PNG of one formula, light on transparent, cached per formula."""
    key = (expr, display)
    if key not in _PNG_CACHE:
        from matplotlib.figure import Figure   # imported on first use, see the note at the top
        fig = Figure(figsize=(1, 1), dpi=200)
        fig.patch.set_alpha(0)
        fig.text(0, 0, f"${expr.strip()}$", fontsize=16 if display else 13, color="#e8eaed")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.03)
        _PNG_CACHE[key] = base64.b64encode(buf.getvalue()).decode()
    return _PNG_CACHE[key]


def render_html(md: str) -> str:
    """Markdown with LaTeX formulas, as HTML a Qt label can show."""
    formulas: list[tuple[str, bool]] = []

    def hold(m: re.Match) -> str:
        expr = m.group(1) if m.group(1) is not None else m.group(2)
        formulas.append((expr, m.group(1) is not None))
        return f"FORMULAHOLD{len(formulas) - 1}X"

    text = _LATEX.sub(hold, md)
    doc = QTextDocument()
    doc.setMarkdown(text)
    out = doc.toHtml()
    for i, (expr, display) in enumerate(formulas):
        try:
            img = (f'<img src="data:image/png;base64,{_formula_png(expr, display)}" '
                   f'height="{30 if display else 18}" style="vertical-align:middle;">')
        except Exception:  # an invalid formula is shown as its source, not dropped
            img = f"<code>{html_lib.escape(expr)}</code>"
        out = out.replace(f"FORMULAHOLD{i}X", img)
    return out


def plain(md: str) -> str:
    """The text a voice should read: no code, no HTML or tool markup, no formatting symbols."""
    t = re.sub(r"```.*?```", " ", md, flags=re.S)              # fenced code is shown, never spoken
    t = re.sub(r"<[^>]+>", " ", t)                              # HTML and tool-call markup
    t = _LATEX.sub(lambda m: (m.group(1) or m.group(2) or "").replace("\\", " "), t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)              # a link is read as its words, not its address
    t = re.sub(r"https?://\S+", " ", t)                         # a bare address is not read out
    t = re.sub(r"^\s*\|?[\s:|-]*-[\s:|-]*\|[\s:|-]*$", " ", t, flags=re.M)     # table separator rows
    t = re.sub(r"^\s*\|(.*)\|\s*$",
               lambda m: ", ".join(c.strip() for c in m.group(1).split("|") if c.strip()) + ".", t, flags=re.M)
    t = re.sub(r"`([^`]*)`", r"\1", t)                          # inline code is spoken as written
    t = re.sub(r"~~|[*_#>]+", "", t)
    t = re.sub(r"^\s*(?:[-+]|\d+\.)\s+", "", t, flags=re.M)
    return re.sub(r"\s+", " ", t).strip()
