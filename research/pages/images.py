"""Edit images: rotate and crop an image file at full resolution (PRD R4).

The browser editor only reports the crop and rotation; OpenCV applies them to
the original file, so nothing is downscaled or re-compressed by the browser.
A copy is saved by default; the original is replaced only after confirmation.
"""

from __future__ import annotations

import base64
import hashlib

import cv2 as cv
import streamlit as st

from research import runtime
from research.catalog import IMAGE_EXTENSIONS, list_images, relative
from research.editing import apply_geometry, is_untouched, read_image, write_image
from ui import style
from ui.components import label_capture
from ui.imaging import to_display_rgb

BROWSER_FORMATS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".bmp": "image/bmp"}
ss = st.session_state


def editor_source(path, image) -> str:
    """A data URL the browser can show. TIFF files are sent as PNG of the same size."""
    mime = BROWSER_FORMATS.get(path.suffix.lower())
    if mime:
        data = path.read_bytes()
    else:
        ok, encoded = cv.imencode(".png", image)
        data, mime = encoded.tobytes(), "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def on_apply() -> None:
    payload = ss["image_editor"].submit
    if isinstance(payload, dict) and isinstance(payload.get("geometry"), dict):
        ss.edit_geometry = {**payload["geometry"], "token": ss.get("edit_token")}


def check_name(name: str, original) -> str | None:
    """An error message for an unusable file name, or None."""
    if not name or name != name.strip() or any(char in name for char in '\\/:*?"<>|'):
        return "Use a plain file name without folders or special characters."
    target = original.with_name(name)
    if target.suffix.lower() not in IMAGE_EXTENSIONS - {".tif", ".tiff", ".bmp"}:
        return "End the name with .png, .jpg, .jpeg or .webp."
    if target.exists():
        return f"{name} already exists. Choose another name."
    return None


# ---------------------------------------------------------------------- page
runtime.page_header("Edit images", "Rotate and crop an image at full resolution. A copy is saved unless you choose to overwrite.")

editor_column, side_column = st.columns([3, 2], gap="large")
with side_column:
    style.section("Image")
    folder = runtime.folder_picker("Folder", key="edit_folder")
    images = list_images(folder) if folder and folder.is_dir() else []
    if not images:
        st.info("Choose a folder that contains images.")
        st.stop()
    path = st.selectbox("Image", images, format_func=lambda item: item.name, key="edit_image")

original = read_image(path)
if original is None:
    side_column.error("This image could not be read.")
    st.stop()
height, width = original.shape[:2]
stat = path.stat()
token = hashlib.sha1(f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode()).hexdigest()[:16]
if ss.get("edit_token") != token:
    ss.edit_token = token
    ss.pop("edit_geometry", None)

with side_column:
    style.facts(
        [
            ("File", relative(path)),
            ("Size", f"{width} × {height} px"),
            ("File size", f"{stat.st_size / 1024:.0f} KB"),
        ]
    )
    if ss.get("edit_flash"):
        st.success(ss.pop("edit_flash"))
with editor_column:
    style.section("Crop & rotate")
    label_capture(
        samples=[],
        image={"token": token, "src": editor_source(path, original), "kind": "file"},
        key="image_editor",
        on_submit=on_apply,
        options={
            "editorOnly": True,
            "geometryOnly": True,
            "submitLabel": "Preview the edit",
            "hint": "Drag the frame or its handles to choose the area to keep.",
        },
    )

geometry = ss.get("edit_geometry")
if geometry and geometry.get("token") == token:
    edited = apply_geometry(original, quarter=int(geometry["quarter"]), fine=float(geometry["fine"]), crop=geometry["crop"])
    with side_column:
        style.section("Preview", f"{edited.shape[1]} × {edited.shape[0]} px")
        st.image(to_display_rgb(edited, max_side=1400), width="stretch")
        if is_untouched(geometry):
            style.note("No change yet: the preview is the original image.")
        name = st.text_input("Save as", value=f"{path.stem}_edited{path.suffix.lower()}", key=f"edit_name_{token}")
        save_clicked = st.button("Save copy", type="primary", width="stretch")
        message = st.empty()  # fixed slot, so messages never shift the widgets below
        with st.expander("Overwrite the original"):
            confirm = st.checkbox(f"Replace {path.name}. This cannot be undone.", key=f"edit_confirm_{token}")
            overwrite_clicked = st.button("Overwrite original", disabled=not confirm, width="stretch")
        if save_clicked:
            problem = check_name(name, path)
            if problem:
                message.error(problem)
            else:
                try:
                    write_image(path.with_name(name), edited)
                except (OSError, ValueError) as error:
                    message.error(f"Could not save: {error}")
                else:
                    message.success(f"Saved {relative(path.with_name(name))}")
        if overwrite_clicked and confirm:
            try:
                write_image(path, edited)
            except (OSError, ValueError) as error:
                message.error(f"Could not overwrite: {error}")
            else:
                ss.pop("edit_geometry", None)
                ss.edit_flash = f"Replaced {relative(path)} with the edited image."
                st.rerun()
else:
    with side_column:
        style.note("Adjust the frame, then press <b>Preview the edit</b>. Nothing is saved until you choose to.")
