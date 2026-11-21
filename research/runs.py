"""Batch runs for the researcher app: run, inspect, and save in the CLI layout.

No Streamlit dependency.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np

import imgproc
from research.catalog import PROJECT_ROOT, relative
from research.editing import read_image
from src.ocr_engine import OCRResult, TextDetector, WordRecognizer, Word, build_result

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"


@dataclass(eq=False)
class ImageRun:
    path: Path
    result: OCRResult | None  # None when the image could not be read
    textmap_png: bytes = b""  # CRAFT region score as a heatmap (same as the CLI's _textmap.png)
    linkmap_png: bytes = b""  # affinity score (refined with the refiner), as _linkmap.png
    error: str | None = None


@dataclass(eq=False)
class Run:
    checkpoint: str  # display name
    settings: dict[str, str]  # what was used, for the log and the saved run
    images: list[ImageRun] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    device: str = ""

    def note(self, message: str) -> None:
        self.log.append(f"[{datetime.now():%H:%M:%S}] {message}")


def _png(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv.imencode(".png", image_bgr)
    return encoded.tobytes() if ok else b""


def run_image(detector: TextDetector, recognizer: WordRecognizer, path: Path) -> ImageRun:
    """Detect and read one image file, keeping what the inspector needs."""
    image = read_image(path)
    if image is None:
        return ImageRun(path, None, error="The image could not be read.")
    detection = detector.detect(image)
    start = time.perf_counter()
    texts, confidences = recognizer.recognize(detection.crops)
    result = build_result(detection, texts, confidences, recognize_seconds=time.perf_counter() - start)
    return ImageRun(
        path,
        result,
        textmap_png=_png(imgproc.cvt2HeatmapImg(detection.score_text)),
        linkmap_png=_png(imgproc.cvt2HeatmapImg(detection.score_link)),
    )


def line_crops(image_bgr: np.ndarray, words: list[Word]) -> list[tuple[int, np.ndarray]]:
    """One crop per reconstructed line (bounding box of its words), as the CLI saves them."""
    height, width = image_bgr.shape[:2]
    crops = []
    for line_number in sorted({word.line for word in words}):
        quads = [word.box for word in words if word.line == line_number]
        left = max(0, int(np.floor(min(quad[:, 0].min() for quad in quads))))
        top = max(0, int(np.floor(min(quad[:, 1].min() for quad in quads))))
        right = min(width - 1, int(np.ceil(max(quad[:, 0].max() for quad in quads))))
        bottom = min(height - 1, int(np.ceil(max(quad[:, 1].max() for quad in quads))))
        if right > left and bottom > top:
            crops.append((line_number, image_bgr[top : bottom + 1, left : right + 1]))
    return crops


def cli_overlay(image_bgr: np.ndarray, result: OCRResult) -> np.ndarray:
    """The CLI's _overlay.jpg: green polygons (refiner) or word boxes."""
    overlay = image_bgr.copy()
    shapes = result.polygons if result.polygons is not None else [word.box for word in result.words]
    for shape in shapes:
        points = np.array(shape, dtype=np.int32).reshape(-1, 1, 2)
        cv.polylines(overlay, [points], True, (0, 255, 0), 2)
    return overlay


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "run"


def save_run(run: Run, base_dir: Path = RUNS_DIR) -> Path:
    """Write the run like the CLI does, into a new folder under ``base_dir``.

    Layout: crops/<stem>_line<L>_word<W>.png, craft/ (heatmaps, overlay,
    lines/, recognized.txt, res_<stem>[_line<L>]_merged.txt), merged.txt and
    run_log.txt. A new folder per run, so nothing earlier is overwritten.
    """
    run_dir = base_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{_slug(run.checkpoint)}"
    crops_dir = run_dir / "crops"
    craft_dir = run_dir / "craft"
    lines_dir = craft_dir / "lines"
    for folder in (crops_dir, lines_dir):
        folder.mkdir(parents=True, exist_ok=False)

    recognized, merged = [], []
    for item in run.images:
        if item.result is None:
            continue
        stem = item.path.stem
        image = read_image(item.path)
        if item.textmap_png:
            (craft_dir / f"{stem}_textmap.png").write_bytes(item.textmap_png)
        if item.linkmap_png:
            (craft_dir / f"{stem}_linkmap.png").write_bytes(item.linkmap_png)
        if image is not None:
            cv.imwrite(str(craft_dir / f"{stem}_overlay.jpg"), cli_overlay(image, item.result))
            for line_number, crop in line_crops(image, item.result.words):
                cv.imwrite(str(lines_dir / f"{stem}_line{line_number}.png"), crop)
        for word in item.result.words:
            crop_path = crops_dir / f"{stem}_line{word.line}_word{word.position}.png"
            cv.imwrite(str(crop_path), word.crop)
            recognized.append(f"{relative(crop_path)}\t{word.text}")
        line_numbers = sorted({word.line for word in item.result.words})  # CLI numbering, gaps kept
        for line_number, line_text in zip(line_numbers, item.result.lines):
            (craft_dir / f"res_{stem}_line{line_number}_merged.txt").write_text(line_text + "\n", encoding="utf-8")
        (craft_dir / f"res_{stem}_merged.txt").write_text("\n".join(item.result.lines) + "\n", encoding="utf-8")
        merged.append(f"{stem}\t{item.result.text}\n\n")

    (craft_dir / "recognized.txt").write_text("".join(line + "\n" for line in recognized), encoding="utf-8")
    (run_dir / "merged.txt").write_text("".join(merged), encoding="utf-8")
    settings = "".join(f"{key}: {value}\n" for key, value in run.settings.items())
    (run_dir / "run_log.txt").write_text(settings + "\n" + "\n".join(run.log) + "\n", encoding="utf-8")
    return run_dir
