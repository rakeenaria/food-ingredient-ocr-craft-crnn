"""Custom Streamlit components (Components v2, inline HTML/CSS/JS)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"


def _asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def _noop() -> None:
    pass


_label_capture = st.components.v2.component(
    "label_capture",
    html=_asset("label_capture.html"),
    css=_asset("label_capture.css"),
    js=_asset("label_capture.js"),
)


def label_capture(
    *,
    samples: list[dict],
    image: dict | None = None,
    notice: dict | None = None,
    key: str = "capture",
    on_sample: Callable[[], None] | None = None,
    on_submit: Callable[[], None] | None = None,
    options: dict | None = None,
):
    """Camera, upload and sample picker with an in-browser crop & rotate editor.

    Triggers: ``sample`` (sample id) and ``submit`` (dict with a JPEG/PNG data URL).
    ``options`` is for the researcher app: ``editorOnly`` shows just the editor,
    ``geometryOnly`` submits the crop and rotation instead of an image, and
    ``submitLabel`` / ``hint`` replace the button and hint texts.
    """
    return _label_capture(
        key=key,
        data={"samples": samples, "image": image, "notice": notice, "options": options or {}},
        on_sample_change=on_sample or _noop,
        on_submit_change=on_submit or _noop,
    )


_ingredient_list = st.components.v2.component(
    "ingredient_list",
    html=_asset("ingredient_list.html"),
    css=_asset("ingredient_list.css"),
    js=_asset("ingredient_list.js"),
)


def ingredient_list(*, items: list[dict], notes: list[dict], palette: dict, key: str = "ingredients") -> None:
    """Numbered ingredient list with an instant, in-browser search.

    ``items`` and ``notes`` are ``{"text", "low", "fixed", "chips", "allergens", "maybe"}``:
    character ranges inside ``text`` of words to check (``[start, end]``), of
    corrected words (``[start, end, word as read]``) and of allergen words
    (``[start, end, group]``, shown in the colors of their group), the groups,
    which the search also matches, and whether the row is a "may contain"
    statement (outlined words). ``palette`` maps each group to its colors.
    """
    _ingredient_list(key=key, data={"items": items, "notes": notes, "palette": palette})


_label_text = st.components.v2.component(
    "label_text",
    html=_asset("label_text.html"),
    css=_asset("label_text.css"),
    js=_asset("label_text.js"),
)


def label_text(
    *,
    text: str,
    marks: list[dict],
    allergens: list[dict],
    palette: dict,
    rejected: list[int],
    editing: bool,
    edit_text: str,
    edit_token: int,
    edited: bool,
    key: str,
    on_rejected: Callable[[], None],
    on_draft: Callable[[], None],
    on_revert: Callable[[], None],
    file_name: str = "ingredients.txt",
):
    """The label text box with copy and download icons, corrections and an editor.

    ``marks`` are ``{"start", "end", "kind", ...}`` ranges inside ``text``:
    ``low`` (to check), ``fixed`` (corrected, with ``id``, ``original`` and
    ``replacement``) or ``undone`` (a correction the user undid).
    ``allergens`` are ``{"start", "end", "group", "maybe"}`` ranges of allergen
    words, shown in the colors of ``palette``. State sent back: ``rejected``
    (ids of undone corrections) and ``draft`` (``{"token", "text"}``, the
    editor's text when it loses focus); trigger: ``revert``.
    """
    return _label_text(
        key=key,
        data={
            "text": text,
            "marks": marks,
            "allergens": allergens,
            "palette": palette,
            "rejected": rejected,
            "editing": editing,
            "editText": edit_text,
            "editToken": edit_token,
            "edited": edited,
            "fileName": file_name,
        },
        on_rejected_change=on_rejected,
        on_draft_change=on_draft,
        on_revert_change=on_revert,
    )
