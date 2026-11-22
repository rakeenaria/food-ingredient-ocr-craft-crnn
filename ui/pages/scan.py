"""Scan: capture or upload a label photo, crop it, and read the ingredient list."""

from __future__ import annotations

import logging

import streamlit as st

from src.allergens import GROUPS, find_allergens, groups_in, summarize
from src.correction import Suggestion, suggest_for_words
from src.postprocess import ComposedText, Span, compose_text, join_wrapped_lines, parse_ingredients
from ui import style
from ui.components import copy_button, ingredient_list, label_capture
from ui.engine import engine_ready, get_engine, log_scan
from ui.imaging import (
    ImageError,
    decode_data_url,
    sample_catalog,
    sample_data_url,
    to_display_rgb,
    word_thumbnail,
)

# Words read with less confidence than this are highlighted (PRD P4). Checked on the
# thesis splits: on the test set it flags 6% of words and catches 12 of the 14 words
# with misread letters or digits; 0.9 catches no more, 0.7 misses two more.
LOW_CONFIDENCE = 0.80
HIGHLIGHT_BGR = (77, 225, 255)  # #FFE14D, the highlighter yellow
logger = logging.getLogger(__name__)

ss = st.session_state
ss.setdefault("stage", "capture")
ss.setdefault("sample_token", 0)
ss.setdefault("pending_image", None)  # image pushed into the capture component
ss.setdefault("submission", None)  # payload received from the capture component
ss.setdefault("notice", None)
ss.setdefault("notice_token", 0)
ss.setdefault("ocr", None)
ss.setdefault("edited_text", None)


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
    # Set, don't pop, the toggle that has a callback: popping drops its key mapping,
    # and a late toggle event from the browser would then call on_toggle_edit with
    # no "editing" key (AttributeError).
    ss.editing = False
    for key in ("show_lines", "edit_area", "photo_view", "word_filter"):
        ss.pop(key, None)
    for key in [key for key in ss if str(key).startswith("fix_")]:  # suggested-fix checkboxes
        ss.pop(key, None)


def current_base_text() -> str:
    return compose_text(ss.ocr["result"].words, as_lines=bool(ss.get("show_lines"))).text


def on_toggle_edit() -> None:
    if not ss.ocr:  # a late toggle event after "Scan another label"
        return
    if ss.get("editing"):
        ss.edit_area = ss.edited_text if ss.edited_text is not None else current_base_text()
    else:
        text = ss.get("edit_area", "")
        changed = text.strip() and text.strip() != current_base_text().strip()
        ss.edited_text = text if changed else None


def undo_edits() -> None:
    ss.edited_text = None


def apply_fixes() -> None:
    """Apply the ticked suggestions to the text shown (PRD P9). Undo brings back the reading."""
    if not ss.ocr:
        return
    chosen = {fix.index: fix.replacement for number, fix in enumerate(suggestions()) if ss.get(f"fix_{number}", True)}
    composed = compose_text(ss.ocr["result"].words, as_lines=bool(ss.get("show_lines")))
    text = composed.text
    for index in sorted(chosen, key=lambda key: composed.spans[key][0] if composed.spans[key] else -1, reverse=True):
        span = composed.spans[index]
        if span is not None:
            text = text[: span[0]] + chosen[index] + text[span[1] :]
    if chosen:
        ss.edited_text = text


def fail(message: str) -> None:
    """Return to the capture step and show ``message`` inside the component."""
    ss.notice_token += 1
    ss.notice = {"token": ss.notice_token, "text": message}
    ss.submission = None
    ss.pending_image = None
    st.rerun()


# ------------------------------------------------------------------ helpers
def low_confidence_marks(words, composed: ComposedText) -> list[style.Mark]:
    """Where the words to check sit in ``composed.text``."""
    return [
        (span[0], span[1], f"Confidence {word.confidence:.0%}. Check it on the label.")
        for word, span in zip(words, composed.spans)
        if span is not None and word.confidence < LOW_CONFIDENCE
    ]


def suggestions() -> list[Suggestion]:
    """Suggested fixes for the words that were probably misread (computed once per scan)."""
    if "suggestions" not in ss.ocr:
        ss.ocr["suggestions"] = suggest_for_words(ss.ocr["result"].words)
    return ss.ocr["suggestions"]


_MARKDOWN_CHARACTERS = set("\\`*_{}[]()#+-.!|<>~")


def md_escape(text: str) -> str:
    """Widget labels are Markdown; OCR text must show literally."""
    return "".join("\\" + char if char in _MARKDOWN_CHARACTERS else char for char in text)


def list_entries(spans: list[Span], marks: list[style.Mark]) -> list[dict]:
    """Ingredient list rows with their low-confidence marks and allergen words."""
    entries = []
    for span in spans:
        inside = [
            [max(start, span.start) - span.start, min(end, span.end) - span.start]
            for start, end, _ in marks
            if start < span.end and end > span.start
        ]
        hits = find_allergens(span.text)
        found = {hit.group for hit in hits}
        entries.append(
            {
                "text": span.text,
                "marks": inside,
                "strong": [[hit.start, hit.end] for hit in hits],
                "allergens": [group for group in GROUPS if group in found],
            }
        )
    return entries


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
    ss.submission = None
    ss.pending_image = None
    ss.notice = None
    ss.edited_text = None
    ss.stage = "results"
    st.rerun()


def render_no_text() -> None:
    style.section("No text found")
    st.html(
        '<ul class="ir-list">'
        "<li>Crop the photo so only the ingredient list is inside.</li>"
        "<li>Use good, even light and avoid glare on glossy packaging.</li>"
        "<li>Hold the phone steady and parallel to the label so the text is sharp.</li>"
        "</ul>"
    )
    st.button("Try another photo", type="primary", on_click=reset, width="stretch")


def render_fixes() -> None:
    """P9: suggested fixes for words that were probably misread, applied only on request."""
    fixes = suggestions()
    if not fixes:
        return
    st.html('<p class="ir-subhead">Possible misreadings</p>')
    for number, fix in enumerate(fixes):
        st.checkbox(f"{md_escape(fix.original)}  →  {md_escape(fix.replacement)}", value=True, key=f"fix_{number}")
    st.button("Apply the ticked fixes", on_click=apply_fixes)
    style.note(
        "These words are not in the ingredient vocabulary but are a letter or two away from a word that is. "
        "Applying them edits the text; <b>Undo my edits</b> brings back what was read."
    )


def render_text(all_words, to_check) -> str | None:
    """The label text with edit, copy and download. Returns the text if the user edited it."""
    style.section("Label text")
    lines_column, edit_column = st.columns(2)
    with lines_column:
        show_lines = st.toggle("Show original lines", key="show_lines")
    with edit_column:
        editing = st.toggle("Edit text", key="editing", on_change=on_toggle_edit)
    composed = compose_text(all_words, as_lines=show_lines)
    edited = None
    if editing:
        st.text_area("Edit the text", key="edit_area", height=240, label_visibility="collapsed")
        final_text = edited = ss.get("edit_area", composed.text)
    elif ss.edited_text is not None:
        style.marked_text(ss.edited_text)
        style.note("You edited this text.")
        st.button("Undo my edits", type="tertiary", on_click=undo_edits)
        final_text = edited = ss.edited_text
    else:
        style.marked_text(composed.text, low_confidence_marks(all_words, composed))
        if to_check:
            style.note("<mark>Highlighted</mark> words were read with low confidence. Check them against the label.")
        final_text = composed.text
        render_fixes()

    copy_column, download_column = st.columns(2)
    with copy_column:
        copy_button(final_text, key="copy_text")
    with download_column:
        st.download_button(
            "Download .txt",
            final_text,
            file_name="ingredients.txt",
            mime="text/plain",
            width="stretch",
        )
    return edited


def render_ingredients(all_words, edited: str | None) -> None:
    """P5: the text split into ingredients, with an instant search."""
    paragraph = compose_text(all_words)
    if edited is not None and join_wrapped_lines(edited) != paragraph.text:
        text, marks = join_wrapped_lines(edited), []
    else:
        text, marks = paragraph.text, low_confidence_marks(all_words, paragraph)
    parsed = parse_ingredients(text)
    count = len(parsed.items)
    style.section("Ingredients", f"{count} item" if count == 1 else f"{count} items")

    # P8: allergens named in the list and its statements, plus ones a suggested fix would reveal.
    if parsed.items:
        summary = summarize([item.text for item in parsed.items], [note.text for note in parsed.notes])
    else:
        summary = summarize([text])
    checks = []
    if edited is None:
        for fix in suggestions():
            new = [g for g in groups_in(fix.replacement) if g not in groups_in(fix.original) and g not in summary.contains]
            if new:
                checks.append((fix.original, fix.replacement, new))
    style.allergen_panel(list(summary.contains), list(summary.may_contain), checks)

    if not parsed.items:
        style.note(
            "No ingredient list was found in this text. Lists usually start with "
            "“Ingredients:” and separate the items with commas."
        )
        return
    ingredient_list(items=list_entries(parsed.items, marks), notes=list_entries(parsed.notes, marks), key="ingredient_list")


def render_photo(image, words, to_check) -> None:
    style.section("Photo")
    view = st.segmented_control(
        "Photo view",
        ["Detected words", "Original"],
        default="Detected words",
        key="photo_view",
        label_visibility="collapsed",
    )
    if view == "Original":
        shown = image
    else:
        from src.ocr_engine import draw_word_boxes  # results exist, so PyTorch is already loaded

        thickness = max(2, round(max(image.shape[:2]) / 700))
        shown = draw_word_boxes(
            image,
            words,
            color=(0, 0, 0),
            thickness=thickness,
            low_confidence_threshold=LOW_CONFIDENCE,
            low_confidence_color=HIGHLIGHT_BGR,
            outline_color=(255, 255, 255),
        )
    st.image(to_display_rgb(shown), width="stretch")
    if view != "Original":
        legend = "Black boxes are the words the reader found."
        if to_check:
            legend += " <mark>Yellow</mark> boxes are words to check."
        style.note(legend)


def render_word_details(words, to_check) -> None:
    """P6: each word crop next to what was read and how sure the reader was."""
    with st.expander("Word details"):
        style.note(
            f"How each word looked to the reader, what it read and how sure it was. "
            f"Words under {LOW_CONFIDENCE:.0%} confidence are <mark>highlighted</mark>."
        )
        choice = "all"
        if to_check:
            labels = {"check": f"Words to check ({len(to_check)})", "all": f"All words ({len(words)})"}
            choice = st.segmented_control(
                "Words to show",
                list(labels),
                format_func=labels.get,
                default="check",
                key="word_filter",
                label_visibility="collapsed",
            )
        low = {id(word) for word in to_check}
        all_words = ss.ocr["result"].words
        fix_for = {id(all_words[fix.index]): fix.replacement for fix in suggestions()}
        rows = [
            (number, thumbnail(word), word.text, word.confidence, id(word) in low, fix_for.get(id(word)))
            for number, word in enumerate(words, start=1)
            if choice == "all" or id(word) in low
        ]
        style.word_table(rows)


def render_results() -> None:
    result = ss.ocr["result"]
    image = ss.ocr["image"]
    style.steps(4)

    words = [word for word in result.words if word.text.strip()]  # in reading order
    if not words:
        render_no_text()
        return
    to_check = [word for word in words if word.confidence < LOW_CONFIDENCE]
    style.facts(
        [
            ("Words read", str(len(words))),
            ("Words to check", str(len(to_check))),
            ("Reading time", f"{result.timings['total']:.1f} s"),
        ]
    )

    edited = render_text(result.words, to_check)
    render_ingredients(result.words, edited)
    render_photo(image, words, to_check)
    render_word_details(words, to_check)

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
