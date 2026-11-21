"""Apply the browser editor's crop & rotate to the full-resolution file.

The editor (ui/assets/label_capture.js) works in "rotated space": the image is
rotated clockwise by ``quarter * 90 + fine`` degrees on a canvas expanded to fit,
and ``crop`` is a rectangle in that space, in original-image pixels. Doing the
same here with OpenCV keeps full resolution and the original file format,
instead of the downscaled JPEG the browser would produce.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2 as cv
import numpy as np

_QUARTER_TURNS = {1: cv.ROTATE_90_CLOCKWISE, 2: cv.ROTATE_180, 3: cv.ROTATE_90_COUNTERCLOCKWISE}
_WRITE_PARAMS = {
    ".jpg": [cv.IMWRITE_JPEG_QUALITY, 95],
    ".jpeg": [cv.IMWRITE_JPEG_QUALITY, 95],
    ".webp": [cv.IMWRITE_WEBP_QUALITY, 95],
    ".png": [cv.IMWRITE_PNG_COMPRESSION, 3],
}


def read_image(path: Path) -> np.ndarray | None:
    """cv.imread that also works with non-ASCII Windows paths (EXIF orientation applied)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv.imdecode(data, cv.IMREAD_COLOR) if data.size else None


def write_image(path: Path, image_bgr: np.ndarray) -> None:
    """Encode by file extension (JPEG/WebP quality 95, PNG lossless) and write."""
    suffix = path.suffix.lower()
    ok, encoded = cv.imencode(suffix, image_bgr, _WRITE_PARAMS.get(suffix, []))
    if not ok:
        raise ValueError(f"Cannot encode an image as {suffix or 'this format'}.")
    path.write_bytes(encoded.tobytes())


def rotated_size(width: int, height: int, degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    cos, sin = abs(math.cos(radians)), abs(math.sin(radians))
    return width * cos + height * sin, width * sin + height * cos


def apply_geometry(image_bgr: np.ndarray, *, quarter: int, fine: float, crop: dict) -> np.ndarray:
    """Rotate clockwise by ``quarter * 90 + fine`` degrees, then crop (white fill)."""
    quarter %= 4
    x, y, w, h = (float(crop[key]) for key in ("x", "y", "w", "h"))
    if fine == 0:
        # Exact quarter turns: no resampling at all.
        rotated = image_bgr if quarter == 0 else cv.rotate(image_bgr, _QUARTER_TURNS[quarter])
        height, width = rotated.shape[:2]
        left = min(max(round(x), 0), width - 1)
        top = min(max(round(y), 0), height - 1)
        right = min(max(round(x + w), left + 1), width)
        bottom = min(max(round(y + h), top + 1), height)
        return rotated[top:bottom, left:right].copy()

    degrees = quarter * 90 + fine
    height, width = image_bgr.shape[:2]
    out_width, out_height = rotated_size(width, height, degrees)
    # OpenCV's positive angle is counter-clockwise; pixel centers sit at integers,
    # so the image center is ((w - 1) / 2, (h - 1) / 2).
    matrix = cv.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), -degrees, 1.0)
    matrix[0, 2] += out_width / 2 - width / 2 - x
    matrix[1, 2] += out_height / 2 - height / 2 - y
    size = (max(1, round(w)), max(1, round(h)))
    return cv.warpAffine(
        image_bgr, matrix, size, flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )


def is_untouched(geometry: dict) -> bool:
    crop = geometry["crop"]
    width, height = rotated_size(geometry["width"], geometry["height"], geometry["quarter"] * 90 + geometry["fine"])
    return (
        geometry["quarter"] % 4 == 0
        and geometry["fine"] == 0
        and crop["x"] <= 0.5
        and crop["y"] <= 0.5
        and abs(crop["w"] - width) <= 1
        and abs(crop["h"] - height) <= 1
    )
