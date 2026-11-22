"""Visual system: a food-label look. White paper, black rules, Libre Franklin.

The only color is the highlighter yellow that marks words to check (PRD D8).
Blocks are rendered with st.html so OCR text is never parsed as Markdown.
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape

import streamlit as st

_CSS = r"""
<style>
[data-testid="stMainBlockContainer"] { max-width: 46rem; padding-top: 2.25rem; padding-bottom: 5rem; }
.ir-masthead { margin: 0.25rem 0 0; }
.ir-title { margin: 0; font-size: clamp(2.3rem, 8vw, 3.4rem); font-weight: 900; line-height: 0.95; letter-spacing: -0.035em; color: #000; }
.ir-lede { margin: 0.75rem 0 0; max-width: 34rem; font-size: 1.05rem; color: #3d3d3a; }
.ir-rule { height: 12px; margin: 1rem 0 1.25rem; border: 0; background: #000; }
.ir-steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 0 0 1.25rem; padding: 0; list-style: none; border-top: 1px solid #111; border-bottom: 1px solid #111; counter-reset: step; font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; }
.ir-steps li { margin: 0; padding: 0.55rem 0.2rem 0.5rem; counter-increment: step; color: #5f5f5a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ir-steps li::before { content: counter(step) "\00a0\00a0"; font-weight: 900; }
.ir-steps li.is-done { color: #111; }
.ir-steps li.is-active { color: #111; font-weight: 800; box-shadow: inset 0 -4px 0 #111; }
.ir-section { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; margin: 1.75rem 0 0.5rem; padding-bottom: 0.35rem; border-bottom: 5px solid #000; font-size: 0.78rem; font-weight: 900; letter-spacing: 0.08em; text-transform: uppercase; }
.ir-section .ir-aside { font-size: 0.82rem; font-weight: 700; letter-spacing: 0.02em; text-transform: none; font-variant-numeric: tabular-nums; }
.ir-facts { width: 100%; margin: 0; border-collapse: collapse; font-size: 0.98rem; }
.ir-facts th, .ir-facts td { padding: 0.45rem 0; border: 0; border-bottom: 1px solid #111; background: transparent; }
.ir-facts th { text-align: left; font-weight: 800; padding-right: 1rem; }
.ir-facts td { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ir-facts td.ir-wrap { white-space: normal; overflow-wrap: anywhere; }
.ir-facts tr:last-child th, .ir-facts tr:last-child td { border-bottom: 0; }
.ir-text { padding: 1rem 1.1rem; border: 2px solid #111; background: #fff; font-size: 1.08rem; line-height: 1.75; overflow-wrap: anywhere; }
.ir-text mark, .ir-note mark, .ir-words mark { background: #ffe14d; color: #000; padding: 0.05em 0.15em; box-shadow: inset 0 -2px 0 #000; }
.ir-note { margin: 0.5rem 0 0; font-size: 0.86rem; color: #4a4a46; }
.ir-list { margin: 0.25rem 0 0; padding-left: 1.3rem; }
.ir-list li { margin: 0.3rem 0; }
.ir-words { width: 100%; margin: 0; border-collapse: collapse; font-size: 0.95rem; }
.ir-words th { padding: 0.3rem 0.5rem 0.3rem 0; border: 0; border-bottom: 2px solid #111; background: transparent; text-align: left; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }
.ir-words td { padding: 0.35rem 0.5rem 0.35rem 0; border: 0; border-bottom: 1px solid #cfcfca; background: transparent; vertical-align: middle; }
.ir-words .n { width: 2.4rem; color: #5f5f5a; font-variant-numeric: tabular-nums; }
.ir-words .crop { width: 42%; }
.ir-words .crop img { display: block; width: auto; max-width: 100%; height: 1.9rem; object-fit: contain; object-position: left center; }
.ir-words .read { font-weight: 600; overflow-wrap: anywhere; }
.ir-words .conf { width: 5.6rem; padding-right: 0; text-align: right; font-variant-numeric: tabular-nums; }
ol.ir-steps { padding: 0 !important; }
[data-testid="stMainBlockContainer"] a { color: #111111; text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 3px; }
[data-testid="stMainBlockContainer"] a:hover { text-decoration-thickness: 2px; }
button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-primary"] { min-height: 2.75rem; border-width: 2px; }
button[data-testid="stBaseButton-secondary"] { border-color: #111111; }
button[data-testid="stBaseButton-secondary"]:hover, button[data-testid="stBaseButton-secondary"]:focus-visible { border-color: #111111; background: #f3f3f1; color: #111111; }
button[data-testid="stBaseButton-secondary"] p, button[data-testid="stBaseButton-primary"] p { font-weight: 700; }
.ir-meters { margin: 0.25rem 0 1rem; }
.ir-meter { display: grid; grid-template-columns: minmax(10rem, 26rem) minmax(6rem, 1fr) 6.5rem; align-items: center; gap: 0.35rem 1rem; padding: 0.45rem 0; border-bottom: 1px solid #e4e4e0; }
.ir-meter-name { font-weight: 700; overflow-wrap: anywhere; }
.ir-meter-track { display: block; height: 10px; background: #e4e4e0; }
.ir-meter-fill { display: block; height: 100%; background: #111111; border-radius: 0 4px 4px 0; }
.ir-meter-figure { text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }
.ir-diff { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.ir-diff .ir-text { font-size: 0.98rem; line-height: 1.65; }
.ir-diff-label { margin: 0 0 0.3rem; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }
@media (max-width: 640px) { .ir-meter { grid-template-columns: 1fr auto; } .ir-meter-name { grid-column: 1 / -1; } .ir-diff { grid-template-columns: 1fr; } }
[data-testid="stExpander"] details { border: 2px solid #111111; }
[data-testid="stExpander"] summary p { font-weight: 800; }
.ir-steps .short, .ir-words .short { display: none; }
@media (max-width: 480px) {
  .ir-steps { font-size: 0.66rem; }
  .ir-steps .long, .ir-words .long { display: none; }
  .ir-steps .short, .ir-words .short { display: inline; }
  .ir-words .n { width: 1.9rem; }
  .ir-words .crop { width: 30%; }
  .ir-words .conf { width: 3.4rem; }
}
</style>
"""

STEP_NAMES = ("Capture", "Crop & rotate", "Read", "Results")
SHORT_STEP_NAMES = ("Capture", "Crop", "Read", "Results")

Mark = tuple[int, int, str]  # (start, end, tooltip) inside a text


def inject(*, wide: bool = False) -> None:
    """Add the styles. ``wide`` lets pages use the full width (researcher app)."""
    st.markdown(_CSS.replace("max-width: 46rem;", "max-width: 104rem;") if wide else _CSS, unsafe_allow_html=True)


def masthead(title: str, lede: str) -> None:
    st.html(
        f'<div class="ir-masthead"><p class="ir-title" role="heading" aria-level="1">{escape(title)}</p>'
        f'<p class="ir-lede">{escape(lede)}</p></div><hr class="ir-rule">'
    )


def steps(active: int) -> None:
    items = []
    for number, (name, short) in enumerate(zip(STEP_NAMES, SHORT_STEP_NAMES), start=1):
        state = "is-active" if number == active else ("is-done" if number < active else "")
        label = escape(name) if name == short else f'<span class="long">{escape(name)}</span><span class="short">{escape(short)}</span>'
        items.append(f'<li class="{state}">{label}</li>')
    st.html(f'<ol class="ir-steps" aria-label="Progress">{"".join(items)}</ol>')


def section(title: str, aside: str | None = None) -> None:
    extra = f'<span class="ir-aside">{escape(aside)}</span>' if aside else ""
    st.html(f'<p class="ir-section"><span>{escape(title)}</span>{extra}</p>')


def facts(rows: list[tuple[str, str]]) -> None:
    """A label-style table. Values stay on one line (numbers never break),
    except long ones such as file paths, which may wrap anywhere."""
    cells = []
    for label, value in rows:
        wrap = ' class="ir-wrap"' if len(value) > 24 or "/" in value or "\\" in value else ""
        cells.append(f'<tr><th scope="row">{escape(label)}</th><td{wrap}>{escape(value)}</td></tr>')
    st.html(f'<table class="ir-facts">{"".join(cells)}</table>')


def note(html_text: str) -> None:
    """Small print. ``html_text`` must already be escaped."""
    st.html(f'<p class="ir-note">{html_text}</p>')


def marked_html(text: str, marks: Iterable[Mark]) -> str:
    """Escaped ``text`` with ``<mark>`` around each range; line breaks become <br>."""
    parts, cursor = [], 0
    for start, end, tooltip in sorted(marks):
        if start < cursor or start >= end:
            continue
        parts.append(escape(text[cursor:start]))
        parts.append(f'<mark title="{escape(tooltip)}">{escape(text[start:end])}</mark>')
        cursor = end
    parts.append(escape(text[cursor:]))
    return "".join(parts).replace("\n", "<br>")


def marked_text(text: str, marks: Iterable[Mark] = ()) -> None:
    """The OCR text in a ruled box, with low-confidence words highlighted."""
    st.html(f'<div class="ir-text">{marked_html(text, marks)}</div>')


def meters(rows: Iterable[tuple[str, float, str]], *, maximum: float = 100.0, unit: str = "%", decimals: int = 3) -> None:
    """A meter list: one row per item with its name, a bar from zero and the figure.

    ``rows`` are (name, value, detail); the detail shows on hover. Pass them in
    the order to show (largest first for a ranking).
    """
    items = []
    for name, value, detail in rows:
        share = max(0.0, min(1.0, value / maximum)) * 100 if maximum else 0.0
        figure = f"{value:.{decimals}f}{unit}"
        items.append(
            f'<div class="ir-meter" role="listitem" title="{escape(detail)}">'
            f'<span class="ir-meter-name">{escape(name)}</span>'
            f'<span class="ir-meter-track" role="img" aria-label="{escape(name)}: {figure}">'
            f'<span class="ir-meter-fill" style="width: {share:.3f}%"></span></span>'
            f'<span class="ir-meter-figure">{figure}</span></div>'
        )
    st.html(f'<div class="ir-meters" role="list">{"".join(items)}</div>')


def text_diff(reference: str, candidate: str, *, reference_label: str, candidate_label: str) -> None:
    """Two texts side by side with the words that differ highlighted."""
    from difflib import SequenceMatcher

    left_words, right_words = reference.split(), candidate.split()
    left, right = [], []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, left_words, right_words, autojunk=False).get_opcodes():
        left_part = escape(" ".join(left_words[i1:i2]))
        right_part = escape(" ".join(right_words[j1:j2]))
        if tag == "equal":
            left.append(left_part)
            right.append(right_part)
            continue
        if left_part:
            left.append(f"<mark>{left_part}</mark>")
        if right_part:
            right.append(f"<mark>{right_part}</mark>")
    st.html(
        '<div class="ir-diff">'
        f'<div><p class="ir-diff-label">{escape(reference_label)}</p><div class="ir-text">{" ".join(left) or "&nbsp;"}</div></div>'
        f'<div><p class="ir-diff-label">{escape(candidate_label)}</p><div class="ir-text">{" ".join(right) or "&nbsp;"}</div></div>'
        "</div>"
    )


def word_table(rows: Iterable[tuple[int, str, str, float, bool]]) -> None:
    """Word details: (number, thumbnail data URL, text, confidence, needs checking)."""
    body = []
    for number, thumbnail, text, confidence, low in rows:
        read = escape(text) or "&nbsp;"
        if low:
            read = f"<mark>{read}</mark>"
        image = f'<img src="{thumbnail}" alt="Word {number} as photographed">' if thumbnail else ""
        body.append(
            f'<tr><td class="n">{number}</td><td class="crop">{image}</td>'
            f'<td class="read">{read}</td><td class="conf">{confidence:.0%}</td></tr>'
        )
    st.html(
        '<table class="ir-words"><thead><tr><th class="n" scope="col">#</th><th scope="col">Photo</th>'
        '<th scope="col">Read as</th><th class="conf" scope="col">'
        '<span class="long">Confidence</span><span class="short">Conf.</span></th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )
