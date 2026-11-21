"""Scan: capture or upload a label photo, crop it, and read the ingredient list."""

from __future__ import annotations

import logging

import streamlit as st

from src.ocr_engine import draw_word_boxes
from src.postprocess import to_lines_text, to_paragraph
from ui import style
from ui.components import copy_button, label_capture
from ui.engine import get_engine
from ui.imaging import ImageError, decode_data_url, sample_catalog, sample_data_url, to_display_rgb

LOW_CONFIDENCE = 0.80  # provisional threshold (PRD P4)
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
    for key in ("show_lines", "editing", "edit_area", "photo_view"):
        ss.pop(key, None)


def current_base_text() -> str:
    lines = ss.ocr["result"].lines
    return to_lines_text(lines) if ss.get("show_lines") else to_paragraph(lines)


def on_toggle_edit() -> None:
    if ss.editing:
        ss.edit_area = ss.edited_text if ss.edited_text is not None else current_base_text()
    else:
        text = ss.get("edit_area", "")
        changed = text.strip() and text.strip() != current_base_text().strip()
        ss.edited_text = text if changed else None


def undo_edits() -> None:
    ss.edited_text = None


def fail(message: str) -> None:
    """Return to the capture step and show ``message`` inside the component."""
    ss.notice_token += 1
    ss.notice = {"token": ss.notice_token, "text": message}
    ss.submission = None
    ss.pending_image = None
    st.rerun()


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
    # Load the models while the visitor frames the photo, so the first scan is quick.
    try:
        get_engine()
    except Exception:
        logger.exception("Could not preload the OCR engine")


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


def render_results() -> None:
    result = ss.ocr["result"]
    image = ss.ocr["image"]
    style.steps(4)

    words = [word for word in result.words if word.text.strip()]
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

    style.section("Ingredients")
    lines_column, edit_column = st.columns(2)
    with lines_column:
        show_lines = st.toggle("Show original lines", key="show_lines")
    with edit_column:
        editing = st.toggle("Edit text", key="editing", on_change=on_toggle_edit)
    base_text = to_lines_text(result.lines) if show_lines else to_paragraph(result.lines)
    if editing:
        st.text_area("Edit the text", key="edit_area", height=240, label_visibility="collapsed")
        final_text = ss.get("edit_area", base_text)
    elif ss.edited_text is not None:
        style.plain_text(ss.edited_text)
        style.note("You edited this text.")
        st.button("Undo my edits", type="tertiary", on_click=undo_edits)
        final_text = ss.edited_text
    else:
        style.highlighted_text(words, LOW_CONFIDENCE, as_lines=show_lines)
        if to_check:
            style.note("<mark>Highlighted</mark> words were read with low confidence. Check them against the label.")
        final_text = base_text

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

    st.html('<hr class="ir-rule">')
    st.button("Scan another label", type="primary", on_click=reset, width="stretch")
    style.note("OCR can make mistakes. Always check the physical label, especially if you have an allergy.")


# ---------------------------------------------------------------------- page
style.inject()
style.masthead(
    "Ingredient Reader",
    "Photograph the ingredient list on a food label and get it back as clean, copyable text.",
)

if ss.stage == "results" and ss.ocr:
    render_results()
elif ss.submission:
    process_submission()
else:
    render_capture()
