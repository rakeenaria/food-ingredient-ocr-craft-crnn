"""Visual system: a food-label look. White paper, black rules, Libre Franklin.

The only color is the highlighter yellow that marks words to check (PRD D8).
Blocks are rendered with st.html so OCR text is never parsed as Markdown.
"""

from __future__ import annotations

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
.ir-section { margin: 1.75rem 0 0.5rem; padding-bottom: 0.35rem; border-bottom: 5px solid #000; font-size: 0.78rem; font-weight: 900; letter-spacing: 0.08em; text-transform: uppercase; }
.ir-facts { width: 100%; margin: 0; border-collapse: collapse; font-size: 0.98rem; }
.ir-facts th, .ir-facts td { padding: 0.45rem 0; border: 0; border-bottom: 1px solid #111; background: transparent; }
.ir-facts th { text-align: left; font-weight: 800; }
.ir-facts td { text-align: right; font-variant-numeric: tabular-nums; }
.ir-facts tr:last-child th, .ir-facts tr:last-child td { border-bottom: 0; }
.ir-text { padding: 1rem 1.1rem; border: 2px solid #111; background: #fff; font-size: 1.08rem; line-height: 1.75; overflow-wrap: anywhere; }
.ir-text mark, .ir-note mark { background: #ffe14d; color: #000; padding: 0.05em 0.15em; box-shadow: inset 0 -2px 0 #000; }
.ir-note { margin: 0.5rem 0 0; font-size: 0.86rem; color: #4a4a46; }
.ir-list { margin: 0.25rem 0 0; padding-left: 1.3rem; }
.ir-list li { margin: 0.3rem 0; }
ol.ir-steps { padding: 0 !important; }
[data-testid="stMainBlockContainer"] a { color: #111111; text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 3px; }
[data-testid="stMainBlockContainer"] a:hover { text-decoration-thickness: 2px; }
button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-primary"] { min-height: 2.75rem; border-width: 2px; }
button[data-testid="stBaseButton-secondary"] { border-color: #111111; }
button[data-testid="stBaseButton-secondary"]:hover, button[data-testid="stBaseButton-secondary"]:focus-visible { border-color: #111111; background: #f3f3f1; color: #111111; }
button[data-testid="stBaseButton-secondary"] p, button[data-testid="stBaseButton-primary"] p { font-weight: 700; }
.ir-steps .short { display: none; }
@media (max-width: 480px) { .ir-steps { font-size: 0.66rem; } .ir-steps .long { display: none; } .ir-steps .short { display: inline; } }
</style>
"""

STEP_NAMES = ("Capture", "Crop & rotate", "Read", "Results")
SHORT_STEP_NAMES = ("Capture", "Crop", "Read", "Results")


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


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


def section(title: str) -> None:
    st.html(f'<p class="ir-section">{escape(title)}</p>')


def facts(rows: list[tuple[str, str]]) -> None:
    body = "".join(f'<tr><th scope="row">{escape(label)}</th><td>{escape(value)}</td></tr>' for label, value in rows)
    st.html(f'<table class="ir-facts">{body}</table>')


def note(html_text: str) -> None:
    """Small print. ``html_text`` must already be escaped."""
    st.html(f'<p class="ir-note">{html_text}</p>')


def highlighted_text(words, threshold: float, *, as_lines: bool) -> None:
    """OCR words in reading order; words below ``threshold`` are highlighted."""
    lines: dict[int, list] = {}
    for word in words:
        lines.setdefault(word.line, []).append(word)
    rendered = []
    for line_number in sorted(lines):
        parts = []
        for word in sorted(lines[line_number], key=lambda w: w.position):
            text = " ".join(word.text.split())
            if not text:
                continue
            if word.confidence < threshold:
                parts.append(f'<mark title="Confidence {word.confidence:.0%}">{escape(text)}</mark>')
            else:
                parts.append(escape(text))
        if parts:
            rendered.append(" ".join(parts))
    separator = "<br>" if as_lines else " "
    st.html(f'<div class="ir-text">{separator.join(rendered)}</div>')


def plain_text(text: str) -> None:
    st.html(f'<div class="ir-text">{escape(text).replace(chr(10), "<br>")}</div>')
