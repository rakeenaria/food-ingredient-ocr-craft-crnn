"""Visual system: a food-label look. White paper, black rules, Libre Franklin.

Color is kept for meaning (PRD D8): a red dotted underline for words to
check, a purple one for words corrected automatically, and one color per
allergen group. Blocks are rendered with st.html so OCR text is never parsed
as Markdown.
"""

from __future__ import annotations

import re
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
.ir-steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 0 0 0.5rem; padding: 0; list-style: none; border-top: 1px solid #111; border-bottom: 1px solid #111; counter-reset: step; font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; }
.ir-steps li { margin: 0; padding: 0.55rem 0.2rem 0.5rem; counter-increment: step; color: #5f5f5a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ir-steps li::before { content: counter(step) "\00a0\00a0"; font-weight: 900; }
.ir-steps li.is-done { color: #111; }
.ir-steps li.is-active { color: #111; font-weight: 800; box-shadow: inset 0 -4px 0 #111; }
.ir-section { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; margin: 1.75rem 0 0.5rem; padding-bottom: 0.35rem; border-bottom: 5px solid #000; font-size: 0.78rem; font-weight: 900; letter-spacing: 0.08em; text-transform: uppercase; }
.ir-section.is-first { margin-top: 0; }
.ir-section .ir-aside { font-size: 0.82rem; font-weight: 700; letter-spacing: 0.02em; text-transform: none; font-variant-numeric: tabular-nums; }
.ir-facts { width: 100%; margin: 0; border-collapse: collapse; font-size: 0.98rem; }
.ir-facts th, .ir-facts td { padding: 0.45rem 0; border: 0; border-bottom: 1px solid #111; background: transparent; }
.ir-facts th { text-align: left; font-weight: 800; padding-right: 1rem; }
.ir-facts td { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ir-facts td.ir-wrap { white-space: normal; overflow-wrap: anywhere; }
.ir-facts tr:last-child th, .ir-facts tr:last-child td { border-bottom: 0; }
.ir-text { padding: 1rem 1.1rem; border: 2px solid #111; background: #fff; font-size: 1.08rem; line-height: 1.75; overflow-wrap: anywhere; }
.ir-text mark, .ir-note mark, .ir-words mark { background: #ffe14d; color: #000; padding: 0.05em 0.15em; box-shadow: inset 0 -2px 0 #000; }
.ir-box-key { display: inline-block; width: 1.5em; height: 0.8em; margin: 0 0.3em 0 0; vertical-align: -0.05em; border: 2px dashed #111; }
.ir-key { white-space: nowrap; }
.st-key-result_photo img { display: block; width: auto !important; height: auto !important; max-width: 100%; max-height: 40vh; margin: 0 auto; }
.st-key-result_photo [data-testid="stFullScreenFrame"] > div { margin-left: auto; margin-right: auto; }
@media (max-width: 360px) {
  .st-key-text_toggles { gap: 0.75rem !important; }
  .st-key-text_toggles [data-testid="stWidgetLabel"] p { font-size: 0.9rem; }
}
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
.ir-allergens { margin: 1.25rem 0 0.25rem; padding: 0.75rem 0.9rem 0.6rem; border: 2px solid #111; }
.ir-allergens-title { margin: 0 0 0.45rem; font-size: 0.78rem; font-weight: 900; letter-spacing: 0.08em; text-transform: uppercase; }
dl.ir-allergens-row { display: grid; grid-template-columns: 7.5rem minmax(0, 1fr); gap: 0.3rem 0.75rem; align-items: baseline; margin: 0 !important; padding: 0.35rem 0 !important; border-top: 1px solid #e4e4e0; }
.ir-allergens-row dt { margin: 0 !important; padding: 0 !important; font-weight: 800; }
.ir-allergens-row dd { margin: 0 !important; padding: 0 !important; display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.3rem; }
.ir-tag { display: inline-block; padding: 0.1rem 0.4rem; background: var(--tag, #111); color: var(--tag-ink, #fff); font-size: 0.74rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; }
.ir-tag.is-maybe { background: #fff; color: #111; box-shadow: inset 0 0 0 2px var(--tag-line, #111); }
.ir-allergens .ir-note { margin-top: 0.45rem; }
.ir-subhead { margin: 1rem 0 0.25rem; font-size: 0.78rem; font-weight: 900; letter-spacing: 0.06em; text-transform: uppercase; }
.ir-words .fix { display: block; margin-top: 0.15rem; font-size: 0.82rem; font-weight: 500; color: #4a4a46; }
@media (max-width: 480px) { .ir-allergens-row { grid-template-columns: 1fr; } }
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

# One color per allergen group, matched to the food: (fill, text on the fill,
# outline for "may contain"). Every text color passes WCAG AA (4.5:1) on its fill.
ALLERGEN_COLORS = {
    "Milk": ("#1F5FBF", "#FFFFFF", "#1F5FBF"),  # milk-carton blue
    "Egg": ("#F2A007", "#111111", "#C98700"),  # yolk
    "Peanut": ("#8A5A2B", "#FFFFFF", "#8A5A2B"),  # roasted peanut
    "Tree nuts": ("#2E7D32", "#FFFFFF", "#2E7D32"),  # green
    "Soy": ("#A3B83A", "#111111", "#7F9420"),  # edamame
    "Wheat / gluten": ("#C99A2E", "#111111", "#A87C1C"),  # ripe wheat
    "Fish": ("#0E7490", "#FFFFFF", "#0E7490"),  # sea blue
    "Shellfish": ("#C8442B", "#FFFFFF", "#C8442B"),  # cooked shrimp
    "Sesame": ("#E8D6A6", "#111111", "#A88B4E"),  # sesame seed
}


def allergen_slug(group: str) -> str:
    """CSS-safe name of an allergen group: "Wheat / gluten" -> "wheat-gluten"."""
    return re.sub(r"[^a-z]+", "-", group.lower()).strip("-")


def allergen_palette() -> dict[str, dict[str, str]]:
    """The allergen colors for the ingredient list component."""
    return {group: {"fill": fill, "ink": ink, "line": line} for group, (fill, ink, line) in ALLERGEN_COLORS.items()}


# Marks on words, in the text, the ingredient list and the word details, and the
# matching box colors on the photo (ui/imaging.py). Both are dotted lines, so the
# color tells them apart; neither is used by an allergen group.
CHECK_COLOR = "#E11D48"  # words to check: red, like a spelling mark
FIXED_COLOR = "#7C3AED"  # words corrected automatically: purple

_ALLERGEN_CSS = " ".join(
    f".ir-tag.a-{allergen_slug(group)} {{ --tag: {fill}; --tag-ink: {ink}; --tag-line: {line}; }}"
    for group, (fill, ink, line) in ALLERGEN_COLORS.items()
)
_MARK_CSS = (
    f".ir-low {{ border-bottom: 2px dotted {CHECK_COLOR}; }} "
    f".ir-fix {{ border-bottom: 2px dotted {FIXED_COLOR}; }} "
    f".ir-box-key.is-check {{ border-color: {CHECK_COLOR}; }} "
    f".ir-box-key.is-fixed {{ border-color: {FIXED_COLOR}; }} "
)
_PAGE_CSS = _CSS.replace("</style>", _ALLERGEN_CSS + " " + _MARK_CSS + "</style>")

STEP_NAMES = ("Capture", "Crop & rotate", "Read", "Results")
SHORT_STEP_NAMES = ("Capture", "Crop", "Read", "Results")


def inject(*, wide: bool = False) -> None:
    """Add the styles. ``wide`` lets pages use the full width (researcher app)."""
    st.markdown(_PAGE_CSS.replace("max-width: 46rem;", "max-width: 104rem;") if wide else _PAGE_CSS, unsafe_allow_html=True)


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


def section(title: str, aside: str | None = None, *, first: bool = False) -> None:
    """A section heading. ``first`` drops the space above it, for the one right under the steps."""
    extra = f'<span class="ir-aside">{escape(aside)}</span>' if aside else ""
    css = "ir-section is-first" if first else "ir-section"
    st.html(f'<p class="{css}"><span>{escape(title)}</span>{extra}</p>')


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


def allergen_panel(contains: list[str], may_contain: list[str], checks: list[tuple[str, str, list[str]]]) -> None:
    """The allergen box above the ingredient list (PRD P8).

    ``checks`` are (word as read, suggested word, groups) for allergens that only
    appear once a suggested fix is applied.
    """
    def tags(groups, maybe=False):
        css = "ir-tag is-maybe" if maybe else "ir-tag"
        return "".join(f'<span class="{css} a-{allergen_slug(group)}">{escape(group)}</span>' for group in groups)

    rows = []
    if contains:
        rows.append(("Contains", tags(contains)))
    if may_contain:
        rows.append(("May contain", tags(may_contain, maybe=True)))
    for original, replacement, groups in checks:
        rows.append(("Check", f"“{escape(original)}” may be “{escape(replacement)}” {tags(groups, maybe=True)}"))
    if rows:
        body = "".join(f'<dl class="ir-allergens-row"><dt>{label}</dt><dd>{value}</dd></dl>' for label, value in rows)
        small = "Found by matching words in the text. Unusual names and misread words can be missed, so check the label."
    else:
        body = '<p class="ir-note" style="margin:0">No common allergens were found in the text.</p>'
        small = "Unusual names and misread words can be missed, so check the label."
    st.html(
        f'<div class="ir-allergens"><p class="ir-allergens-title">Allergens</p>{body}'
        f'<p class="ir-note">{escape(small)}</p></div>'
    )


def word_table(rows: Iterable[tuple], *, low_style: str = "mark") -> None:
    """Word details: (number, thumbnail data URL, text, confidence, needs checking[, note]).

    ``note`` is ``("fixed", word)`` for a word corrected automatically or
    ``("suggested", word)`` for a correction the user undid. ``low_style``
    "underline" gives words to check a red dotted underline (public app) instead
    of the yellow highlight (researcher app).
    """
    body = []
    for number, thumbnail, text, confidence, low, *extra in rows:
        read = escape(text) or "&nbsp;"
        if low:
            read = f'<span class="ir-low">{read}</span>' if low_style == "underline" else f"<mark>{read}</mark>"
        note = extra[0] if extra else None
        if note:
            kind, word = note
            if kind == "fixed":
                read += f'<span class="fix">Corrected to <span class="ir-fix">{escape(word)}</span></span>'
            else:
                read += f'<span class="fix">Suggested: {escape(word)}</span>'
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
