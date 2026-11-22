"""Image decoding, display helpers and the bundled sample photos."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Collection
from pathlib import Path

import cv2 as cv
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # not imported from src: that would load PyTorch
SAMPLES_DIR = PROJECT_ROOT / "images" / "gambar_simulasi"
SAMPLE_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_SIDE = 4096  # defensive cap; the browser already limits edited images to 2560 px
MIN_SIDE = 32


class ImageError(ValueError):
    """A user-facing problem with a submitted image."""


def decode_data_url(data_url: object) -> np.ndarray:
    """Decode a ``data:image/...;base64,`` URL into an upright BGR array."""
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise ImageError("The photo could not be read. Please try again.")
    header, separator, payload = data_url.partition(",")
    if not separator or ";base64" not in header:
        raise ImageError("The photo could not be read. Please try again.")
    if len(payload) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise ImageError("This photo is too large. The limit is 20 MB.")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ImageError("The photo could not be read. Please try again.") from error

    # imdecode applies EXIF orientation, so phone photos come out upright.
    image = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ImageError("This image format is not supported. Please use JPG, PNG or WebP.")
    height, width = image.shape[:2]
    if min(height, width) < MIN_SIDE:
        raise ImageError("This photo is too small to read.")
    if max(height, width) > MAX_SIDE:
        scale = MAX_SIDE / max(height, width)
        image = cv.resize(image, (round(width * scale), round(height * scale)), interpolation=cv.INTER_AREA)
    return image


def to_display_rgb(image_bgr: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """RGB copy for st.image, downscaled so the browser gets a light image."""
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image_bgr = cv.resize(image_bgr, (round(width * scale), round(height * scale)), interpolation=cv.INTER_AREA)
    return cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)


INK_BGR = (17, 17, 17)
PAPER_BGR = (255, 255, 255)
CHECK_BGR = (72, 29, 225)  # #E11D48, style.CHECK_COLOR: words to check
FIXED_BGR = (237, 58, 124)  # #7C3AED, style.FIXED_COLOR: words corrected automatically


def _dashed_polygon(canvas: np.ndarray, points: np.ndarray, color, thickness: int, dash: float, gap: float) -> None:
    """Draw a closed polygon as dashes; every edge starts with a dash so corners show."""
    for i in range(len(points)):
        start = points[i].astype(np.float64)
        end = points[(i + 1) % len(points)].astype(np.float64)
        length = float(np.hypot(*(end - start)))
        if length < 1:
            continue
        step = (end - start) / length
        position = 0.0
        while position < length:
            a = start + step * position
            b = start + step * min(position + dash, length)
            cv.line(canvas, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), color, thickness, cv.LINE_AA)
            position += dash + gap


def draw_result_boxes(image_bgr: np.ndarray, words, *, check: Collection[int] = (), fixed: Collection[int] = ()) -> np.ndarray:
    """A copy of the photo with a box around each word read.

    Words to check and corrected words get a dashed box in the color of their
    dotted underline in the text: red and purple. ``check`` and ``fixed`` are
    indices into ``words``. A light halo keeps boxes visible on dark packaging.
    """
    canvas = image_bgr.copy()
    longest = max(canvas.shape[:2])
    thickness = max(2, round(longest / 700))
    dash, gap = max(8.0, longest / 70), max(5.0, longest / 120)
    for index, word in enumerate(words):
        if not word.text.strip():
            continue
        points = np.round(word.box).astype(np.int32)
        if index in fixed or index in check:
            _dashed_polygon(canvas, points, PAPER_BGR, thickness + 2, dash, gap)
            _dashed_polygon(canvas, points, FIXED_BGR if index in fixed else CHECK_BGR, thickness, dash, gap)
        else:
            cv.polylines(canvas, [points.reshape(-1, 1, 2)], True, PAPER_BGR, thickness + 2, cv.LINE_AA)
            cv.polylines(canvas, [points.reshape(-1, 1, 2)], True, INK_BGR, thickness, cv.LINE_AA)
    return canvas


def word_thumbnail(crop_bgr: np.ndarray, height: int = 48, max_width: int = 480) -> str:
    """A small JPEG data URL of a word crop for the word details table."""
    crop_height, crop_width = crop_bgr.shape[:2]
    scale = min(height / crop_height, max_width / crop_width)
    if scale < 1.0:
        size = (max(1, round(crop_width * scale)), max(1, round(crop_height * scale)))
        crop_bgr = cv.resize(crop_bgr, size, interpolation=cv.INTER_AREA)
    ok, encoded = cv.imencode(".jpg", crop_bgr, [cv.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _sample_paths() -> list:
    return [path for path in sorted(SAMPLES_DIR.glob("*")) if path.suffix.lower() in SAMPLE_MIME]


@st.cache_data(show_spinner=False)
def sample_catalog() -> list[dict]:
    """Thumbnails of the bundled sample labels for the capture component."""
    catalog = []
    for index, path in enumerate(_sample_paths(), start=1):
        image = cv.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        thumb_width = 240
        thumb = cv.resize(image, (thumb_width, max(1, round(height * thumb_width / width))), interpolation=cv.INTER_AREA)
        ok, encoded = cv.imencode(".jpg", thumb, [cv.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        catalog.append(
            {
                "id": path.name,
                "label": f"sample label {index}",
                "thumb": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
        )
    return catalog


@st.cache_data(show_spinner=False)
def sample_data_url(sample_id: str) -> str | None:
    """Original bytes of a bundled sample as a data URL (only known sample ids)."""
    for path in _sample_paths():
        if path.name == sample_id:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{SAMPLE_MIME[path.suffix.lower()]};base64,{encoded}"
    return None
