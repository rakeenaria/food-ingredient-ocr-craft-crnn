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
):
    """Camera, upload and sample picker with an in-browser crop & rotate editor.

    Triggers: ``sample`` (sample id) and ``submit`` (dict with a JPEG/PNG data URL).
    """
    return _label_capture(
        key=key,
        data={"samples": samples, "image": image, "notice": notice},
        on_sample_change=on_sample or _noop,
        on_submit_change=on_submit or _noop,
    )


_ingredient_list = st.components.v2.component(
    "ingredient_list",
    html=_asset("ingredient_list.html"),
    css=_asset("ingredient_list.css"),
    js=_asset("ingredient_list.js"),
)


def ingredient_list(*, items: list[dict], notes: list[dict], key: str = "ingredients") -> None:
    """Numbered ingredient list with an instant, in-browser search.

    ``items`` and ``notes`` are ``{"text": str, "marks": [[start, end], ...]}``;
    marks are character ranges of low-confidence words inside ``text``.
    """
    _ingredient_list(key=key, data={"items": items, "notes": notes})


_COPY_HTML ='<button type="button" class="copy"><span class="label">Copy text</span></button>'

_COPY_CSS = """
.copy {
  appearance: none;
  width: 100%;
  min-height: 2.75rem;
  padding: 0.55rem 1rem;
  border: 2px solid #111111;
  border-radius: 0;
  background: #ffffff;
  color: #111111;
  font-family: var(--st-font, "Libre Franklin", sans-serif);
  font-size: 0.95rem;
  font-weight: 700;
  cursor: pointer;
}
.copy:hover { background: #f3f3f1; }
.copy.is-done { background: #111111; color: #ffffff; }
.copy:focus-visible { outline: 3px solid #111111; outline-offset: 2px; }
"""

_COPY_JS = """
export default function ({ parentElement, data }) {
  const button = parentElement.querySelector("button");
  const label = button.querySelector(".label");
  const idle = (data && data.label) || "Copy text";
  if (!button.classList.contains("is-done")) label.textContent = idle;
  button.onclick = async () => {
    const text = (data && data.text) || "";
    let copied = false;
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
    } catch (_) {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try { copied = document.execCommand("copy"); } catch (_) { copied = false; }
      area.remove();
    }
    label.textContent = copied ? "Copied" : "Copy failed";
    button.classList.toggle("is-done", copied);
    clearTimeout(button._resetTimer);
    button._resetTimer = setTimeout(() => {
      label.textContent = idle;
      button.classList.remove("is-done");
    }, 1600);
  };
}
"""

_copy_button = st.components.v2.component("copy_text_button", html=_COPY_HTML, css=_COPY_CSS, js=_COPY_JS)


def copy_button(text: str, *, key: str, label: str = "Copy text") -> None:
    """A button that copies ``text`` to the clipboard (client-side)."""
    _copy_button(key=key, data={"text": text, "label": label})
