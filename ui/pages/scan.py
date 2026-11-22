"""Scan: capture or upload a label photo, crop it, and read the ingredient list."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import streamlit as st

from src.allergens import (
    GROUPS,
    AllergenSummary,
    find_allergens,
    find_in_list,
    groups_in,
    is_precautionary,
    summarize,
)
from src.correction import Suggestion, apply_suggestions, suggest_for_words
from src.postprocess import (
    ComposedText,
    IngredientList,
    Span,
    compose_text,
    join_wrapped_lines,
    parse_ingredients,
    remap_spans,
)
from ui import style
from ui.components import ingredient_list, label_capture, label_text
from ui.engine import engine_ready, get_engine, log_scan
from ui.imaging import (
    ImageError,
    decode_data_url,
    draw_result_boxes,
    sample_catalog,
    sample_data_url,
    to_display_rgb,
    word_thumbnail,
)

# Words read with less confidence than this get a yellow dotted underline (PRD P4). Checked
# on the thesis splits: on the test set it flags 6% of words and catches 12 of the 14
# words with misread letters or digits; 0.9 catches no more, 0.7 misses two more.
LOW_CONFIDENCE = 0.80
logger = logging.getLogger(__name__)

ss = st.session_state
ss.setdefault("stage", "capture")
ss.setdefault("sample_token", 0)
ss.setdefault("pending_image", None)  # image pushed into the capture component
ss.setdefault("submission", None)  # payload received from the capture component
ss.setdefault("notice", None)
ss.setdefault("notice_token", 0)
ss.setdefault("ocr", None)
ss.setdefault("scan_id", 0)  # every scan gets a fresh label text component
ss.setdefault("edited_text", None)  # the user's own edit of the text
ss.setdefault("rejected_fixes", frozenset())  # words whose automatic correction was undone
ss.setdefault("edit_token", 0)  # changes each time editing starts
ss.setdefault("edit_start", "")  # the text the editor starts with


# ----------------------------------------------------------------- callbacks
def on_sample() -> None:
    sample_id = ss["capture"].sample
    src = sample_data_url(sample_id) if isinstance(sample_id, str) else None
    if src is None:
        return
    ss.sample_token += 1
    ss.pending_image = {"token": ss.sample_token, "src": src, "kind": "sample"}


def on_submit() -> None:
    ss.submission = ss["capture"].submit


def reset() -> None:
    ss.stage = "capture"
    ss.ocr = None
    ss.submission = None
    ss.pending_image = None
    ss.notice = None
    ss.edited_text = None
    ss.rejected_fixes = frozenset()
    ss.edit_start = ""
    # Set, don't pop, the toggle that has a callback: popping drops its key mapping,
    # and a late toggle event from the browser would then call on_toggle_edit with
    # no "editing" key (AttributeError).
    ss.editing = False
    for key in ("show_lines", "photo_view", "word_filter"):
        ss.pop(key, None)


def text_key() -> str:
    return f"label_text_{ss.scan_id}"


def text_state(name: str):
    """A state value the label text component sent back."""
    value = ss.get(text_key())
    return value.get(name) if isinstance(value, dict) else None


def auto_text() -> str:
    """The text as shown before any edit: corrected words, as lines or as a paragraph."""
    words = apply_suggestions(ss.ocr["result"].words, suggestions(), skip=ss.rejected_fixes)
    return compose_text(words, as_lines=bool(ss.get("show_lines"))).text


def on_toggle_edit() -> None:
    if not ss.ocr:  # a late toggle event after "Scan another label"
        return
    if ss.get("editing"):
        ss.edit_token += 1
        ss.edit_start = ss.edited_text if ss.edited_text is not None else auto_text()


def on_rejected() -> None:
    """Undo or redo of an automatic correction (PRD P9)."""
    if not ss.ocr:
        return
    value = text_state("rejected")
    ids = value if isinstance(value, list) else []
    ss.rejected_fixes = frozenset(int(item) for item in ids if isinstance(item, (int, float)))


def on_draft() -> None:
    """The editor lost focus: keep its text if it differs from the text as read."""
    if not ss.ocr:
        return
    draft = text_state("draft")
    if not isinstance(draft, dict) or draft.get("token") != ss.edit_token:
        return  # from an earlier edit
    text = str(draft.get("text") or "")
    ss.edited_text = text if text.strip() and text.strip() != auto_text().strip() else None


def on_revert() -> None:
    ss.edited_text = None


def fail(message: str) -> None:
    """Return to the capture step and show ``message`` inside the component."""
    ss.notice_token += 1
    ss.notice = {"token": ss.notice_token, "text": message}
    ss.submission = None
    ss.pending_image = None
    st.rerun()


# ------------------------------------------------------------------ helpers
def suggestions() -> list[Suggestion]:
    """Corrections for the words that were probably misread (computed once per scan)."""
    if "suggestions" not in ss.ocr:
        ss.ocr["suggestions"] = suggest_for_words(ss.ocr["result"].words)
    return ss.ocr["suggestions"]


@dataclass
class Reading:
    """The scan with the user's choices applied, worked out once per run."""

    words: list  # words as read, in reading order
    corrected: list  # the same words with the corrections that were kept
    fixes: dict[int, Suggestion]  # word index -> suggested correction
    fixed: set[int]  # corrected words (green dotted underline)
    check: set[int]  # words to check (yellow dotted underline): low confidence or a correction undone
    seconds: float


def current_reading() -> Reading:
    result = ss.ocr["result"]
    fixes = {fix.index: fix for fix in suggestions()}
    undone = ss.rejected_fixes & fixes.keys()
    fixed = set(fixes) - undone
    check = {
        index
        for index, word in enumerate(result.words)
        if word.text.strip() and index not in fixed and (word.confidence < LOW_CONFIDENCE or index in undone)
    }
    corrected = apply_suggestions(result.words, list(fixes.values()), skip=undone)
    return Reading(result.words, corrected, fixes, fixed, check, result.timings["total"])


def text_marks(reading: Reading, composed: ComposedText) -> list[dict]:
    """Where the corrected words and the words to check sit in ``composed.text``."""
    marks = []
    for index, span in enumerate(composed.spans):
        if span is None:
            continue
        start, end = span
        if index in reading.fixes:
            fix = reading.fixes[index]
            kind = "fixed" if index in reading.fixed else "undone"
            marks.append(
                {"start": start, "end": end, "kind": kind, "id": index, "original": fix.original, "replacement": fix.replacement}
            )
        elif index in reading.check:
            tip = f"Confidence {reading.words[index].confidence:.0%}. Check it on the label."
            marks.append({"start": start, "end": end, "kind": "low", "tip": tip})
    return marks


@dataclass
class Analysis:
    """The ingredient list and allergens of the text shown."""

    text: str  # the paragraph that was parsed
    parsed: IngredientList
    low: list[tuple[int, int]]  # words to check, inside the parsed text
    fixed: list[tuple[int, int, str]]  # corrected words with the word as read
    allergens: AllergenSummary
    checks: list[tuple[str, str, list[str]]]  # undone corrections that would name an allergen


def analyse(reading: Reading, edited: str | None) -> Analysis:
    """P5 and P8 on the corrected text, or on the user's edit."""
    paragraph = compose_text(reading.corrected)
    if edited is not None and join_wrapped_lines(edited) != paragraph.text:
        text, low, fixed = join_wrapped_lines(edited), [], []
    else:
        text = paragraph.text
        low = [paragraph.spans[index] for index in sorted(reading.check) if paragraph.spans[index]]
        fixed = [
            (*paragraph.spans[index], reading.fixes[index].original)
            for index in sorted(reading.fixed)
            if paragraph.spans[index]
        ]
    parsed = parse_ingredients(text)
    if parsed.items:
        allergens = summarize([item.text for item in parsed.items], [note.text for note in parsed.notes])
    else:
        allergens = summarize([text])
    checks = []
    if edited is None:
        for index, fix in reading.fixes.items():
            if index in reading.fixed:
                continue  # applied, so already in the summary
            new = [g for g in groups_in(fix.replacement) if g not in groups_in(fix.original) and g not in allergens.contains]
            if new:
                checks.append((fix.original, fix.replacement, new))
    return Analysis(text, parsed, low, fixed, allergens, checks)


def text_allergens(analysis: Analysis, shown: str) -> list[dict]:
    """Where the allergen words sit in the text shown, and whether only "may contain" (P8).

    The words come from the same ingredients and statements as the allergen box,
    found in the parsed paragraph and moved onto ``shown``, which can differ in
    line breaks (the label's lines, or the user's edit).
    """
    parsed = analysis.parsed
    if parsed.items:
        ranges = lambda spans: [(span.start, span.end) for span in spans]  # noqa: E731
        hits = find_in_list(analysis.text, ranges(parsed.items), ranges(parsed.notes))
    else:
        hits = find_allergens(analysis.text)  # no list found: the box reads the whole text too
    moved = remap_spans([(hit.start, hit.end) for hit in hits], analysis.text, shown)
    return [
        {"start": span[0], "end": span[1], "group": hit.group, "maybe": hit.precaution}
        for hit, span in zip(hits, moved)
        if span is not None
    ]


def list_entries(
    spans: list[Span], low: list[tuple[int, int]], fixed: list[tuple[int, int, str]], *, notes: bool = False
) -> list[dict]:
    """Ingredient list rows with their words to check, corrected words and allergens.

    Allergen words get the colors of their group, as in the label text, and each row
    tags its groups. For ``notes``, a "may contain" statement gets outlined words and
    tags, as in the allergen box.
    """
    entries = []
    for span in spans:
        hits = find_allergens(span.text)
        found = {hit.group for hit in hits}
        entries.append(
            {
                "text": span.text,
                "low": [
                    [max(start, span.start) - span.start, min(end, span.end) - span.start]
                    for start, end in low
                    if start < span.end and end > span.start
                ],
                "fixed": [
                    [max(start, span.start) - span.start, min(end, span.end) - span.start, original]
                    for start, end, original in fixed
                    if start < span.end and end > span.start
                ],
                "chips": [[hit.start, hit.end, hit.group] for hit in hits],
                "allergens": [group for group in GROUPS if group in found],
                "maybe": notes and is_precautionary(span.text),
            }
        )
    return entries


def plural(count: int, word: str) -> str:
    return f"{count} {word if count == 1 else word + 's'}"


def thumbnail(word) -> str:
    cache = ss.ocr.setdefault("thumbnails", {})
    if id(word) not in cache:
        cache[id(word)] = word_thumbnail(word.crop)
    return cache[id(word)]


# --------------------------------------------------------------------- views
def render_capture() -> None:
    label_capture(
        samples=sample_catalog(),
        image=ss.pending_image,
        notice=ss.notice,
        key="capture",
        on_sample=on_sample,
        on_submit=on_submit,
    )
    style.note("Works best with printed English ingredient lists. Photos are processed in memory and never stored.")
    # The models are already loading in the background (app_streamlit.py), so nothing waits here.


def process_submission() -> None:
    payload = ss.submission if isinstance(ss.submission, dict) else {}
    style.steps(3)
    try:
        image = decode_data_url(payload.get("image"))
    except ImageError as error:
        fail(str(error))
        return

    st.image(to_display_rgb(image, max_side=1000), width="stretch")
    with st.status("Reading the label…", expanded=True) as status:
        if not engine_ready():
            status.write("Getting the reader ready. After a restart this takes a little longer…")
        try:
            engine = get_engine()
        except Exception:
            logger.exception("Could not load the OCR engine")
            status.update(label="The reader is unavailable", state="error", expanded=False)
            fail("The reader is unavailable right now. Please try again in a few minutes.")
            return
        if engine.busy:
            status.write("Another scan is in progress. Yours starts next…")

        def progress(stage: str, count: int) -> None:
            if stage == "detect":
                status.write("Finding the text on the label…")
            elif stage == "recognize":
                status.write(f"Reading {count} words…")

        try:
            result = engine.process(image, on_progress=progress)
        except Exception:
            logger.exception("OCR failed")
            status.update(label="Something went wrong", state="error", expanded=False)
            fail("Something went wrong while reading this photo. Please try again or use another photo.")
            return
        status.update(label=f"Done in {result.timings['total']:.1f} s", state="complete", expanded=False)
    log_scan(result.timings["total"], len(result.words), result.image_size)

    ss.ocr = {"result": result, "image": image}
    ss.scan_id += 1
    ss.submission = None
    ss.pending_image = None
    ss.notice = None
    ss.edited_text = None
    ss.rejected_fixes = frozenset()
    ss.stage = "results"
    st.rerun()


def render_no_text() -> None:
    style.section("No text found", first=True)
    st.html(
        '<ul class="ir-list">'
        "<li>Crop the photo so only the ingredient list is inside.</li>"
        "<li>Use good, even light and avoid glare on glossy packaging.</li>"
        "<li>Hold the phone steady and parallel to the label so the text is sharp.</li>"
        "</ul>"
    )
    st.button("Try another photo", type="primary", on_click=reset, width="stretch")


def render_photo(image, reading: Reading) -> None:
    """The photo first, kept short, with the words found boxed (P6)."""
    style.section("Photo", first=True)
    view = st.segmented_control(
        "Photo view",
        ["Detected words", "Original"],
        default="Detected words",
        key="photo_view",
        label_visibility="collapsed",
    )
    detected = view != "Original"
    shown = draw_result_boxes(image, reading.words, check=reading.check, fixed=reading.fixed) if detected else image
    with st.container(key="result_photo"):
        st.image(to_display_rgb(shown), width="stretch")

    count = sum(1 for word in reading.words if word.text.strip())
    summary = f"<b>{plural(count, 'word')}</b> read in {reading.seconds:.1f} s."
    if detected:
        keys = []
        if reading.check:
            keys.append(f'<span class="ir-key"><span class="ir-box-key is-check"></span>{len(reading.check)} to check</span>')
        if reading.fixed:
            keys.append(f'<span class="ir-key"><span class="ir-box-key is-fixed"></span>{len(reading.fixed)} corrected</span>')
        if keys:
            summary += " &nbsp;" + " &nbsp; ".join(keys)
    else:
        details = [f"{len(reading.check)} to check"] if reading.check else []
        details += [f"{len(reading.fixed)} corrected"] if reading.fixed else []
        if details:
            summary += " " + ", ".join(details).capitalize() + "."
    style.note(summary)


def render_text(reading: Reading, edited: str | None, analysis: Analysis) -> None:
    """The label text with allergens, corrections, copy, download and editing (P3, P4, P8, P9)."""
    style.section("Label text")
    with st.container(horizontal=True, gap="medium", key="text_toggles"):
        show_lines = st.toggle("Show original lines", key="show_lines")
        editing = st.toggle("Edit text", key="editing", on_change=on_toggle_edit)
    composed = compose_text(reading.corrected, as_lines=show_lines)
    if editing or edited is not None:
        text, marks = (edited if edited is not None else composed.text), []
    else:
        text, marks = composed.text, text_marks(reading, composed)
    label_text(
        text=text,
        marks=marks,
        allergens=[] if editing else text_allergens(analysis, text),
        palette=style.allergen_palette(),
        rejected=sorted(set(reading.fixes) - reading.fixed),
        editing=editing,
        edit_text=ss.edit_start,
        edit_token=ss.edit_token,
        edited=edited is not None,
        key=text_key(),
        on_rejected=on_rejected,
        on_draft=on_draft,
        on_revert=on_revert,
    )
    if editing or edited is not None:
        return
    legend = []
    if reading.check:
        legend.append('<span class="ir-low">Yellow dotted</span> words were read with low confidence. Check them on the label.')
    if reading.fixed:
        legend.append(
            '<span class="ir-fix">Green dotted</span> words were corrected automatically. '
            "Hover or tap one to see what was read, or to undo it."
        )
    if legend:
        style.note(" ".join(legend))


def render_ingredients(analysis: Analysis) -> None:
    """P5: the text split into ingredients, with an instant search."""
    parsed = analysis.parsed
    count = len(parsed.items)
    style.section("Ingredients", plural(count, "item"))
    if not parsed.items:
        style.note(
            "No ingredient list was found in this text. Lists usually start with "
            "“Ingredients:” and separate the items with commas."
        )
        return
    ingredient_list(
        items=list_entries(parsed.items, analysis.low, analysis.fixed),
        notes=list_entries(parsed.notes, analysis.low, analysis.fixed, notes=True),
        palette=style.allergen_palette(),
        key="ingredient_list",
    )


def render_word_details(reading: Reading) -> None:
    """P6: each word crop next to what was read and how sure the reader was."""
    with st.expander("Word details"):
        style.note(
            "How each word looked to the reader, what it read and how sure it was. Flagged words were "
            f'read with less than {LOW_CONFIDENCE:.0%} confidence (<span class="ir-low">yellow dotted underline</span>) '
            'or <span class="ir-fix">corrected automatically</span>.'
        )
        numbered = [(index, word) for index, word in enumerate(reading.words) if word.text.strip()]
        flagged = reading.check | reading.fixed
        choice = "all"
        if flagged:
            labels = {"flagged": f"Flagged ({len(flagged)})", "all": f"All words ({len(numbered)})"}
            choice = st.segmented_control(
                "Words to show",
                list(labels),
                format_func=labels.get,
                default="flagged",
                key="word_filter",
                label_visibility="collapsed",
            )
        rows = []
        for number, (index, word) in enumerate(numbered, start=1):
            if choice != "all" and index not in flagged:
                continue
            note = None
            if index in reading.fixes:
                note = ("fixed" if index in reading.fixed else "suggested", reading.fixes[index].replacement)
            rows.append((number, thumbnail(word), word.text, word.confidence, index in reading.check, note))
        style.word_table(rows, low_style="underline")


def render_results() -> None:
    result = ss.ocr["result"]
    style.steps(4)
    if not any(word.text.strip() for word in result.words):
        render_no_text()
        return

    reading = current_reading()
    edited = ss.edited_text
    analysis = analyse(reading, edited)

    # What matters most comes first: the photo, the allergens, then the text.
    render_photo(ss.ocr["image"], reading)
    style.allergen_panel(list(analysis.allergens.contains), list(analysis.allergens.may_contain), analysis.checks)
    render_text(reading, edited, analysis)
    render_ingredients(analysis)
    render_word_details(reading)

    st.html('<hr class="ir-rule">')
    st.button("Scan another label", type="primary", on_click=reset, width="stretch")
    style.note("OCR can make mistakes. Always check the physical label, especially if you have an allergy.")


# ---------------------------------------------------------------------- page
style.inject()
style.masthead(
    "Ingredient Facts",
    "Photograph the ingredient list on a food label. Get it back as clean text and a searchable list.",
)

if ss.stage == "results" and ss.ocr:
    render_results()
elif ss.submission:
    process_submission()
else:
    render_capture()
