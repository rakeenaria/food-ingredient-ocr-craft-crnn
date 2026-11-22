"""Process-wide OCR engine for the Streamlit app, loaded in the background.

The first page render must not wait for PyTorch: importing it and loading the
two models takes several seconds on a small CPU host. ``start_loading()`` (called
by the router on every page) starts that work on a background thread, once per
server process, while the visitor frames the photo. ``get_engine()`` waits for
it only when a scan actually needs the engine.

Locally the weights come from saved_models/. They are not in git, so on the
hosting platform they are downloaded once from the public Hugging Face model
repo (no token needed) and kept in the Hugging Face cache. See docs/DEPLOY.md.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # importing src.ocr_engine pulls in PyTorch; only do it on the loader thread
    from src.ocr_engine import OCREngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WARMUP_IMAGE = PROJECT_ROOT / "images" / "gambar_simulasi" / "04.jpeg"
# Public model repo holding best_accuracy.pth, opt.txt and craft_mlt_25k.pth.
WEIGHTS_REPO = os.environ.get("INGREDIENT_FACTS_WEIGHTS_REPO", "rakeenarya/ingredient-reader")
WEIGHTS_REVISION = os.environ.get("INGREDIENT_FACTS_WEIGHTS_REVISION", "main")
LOAD_TIMEOUT_SECONDS = 600

_lock = threading.Lock()
_finished = threading.Event()
_loader: threading.Thread | None = None
_engine: OCREngine | None = None
_error: BaseException | None = None


def weight_paths(*, prefer_local: bool = True) -> tuple[Path, Path]:
    """(recognizer, detector): the local files when present, otherwise the Hub copies."""
    from src.ocr_engine import DEFAULT_DETECTOR_PATH, DEFAULT_RECOGNIZER_PATH

    if prefer_local and DEFAULT_RECOGNIZER_PATH.exists() and DEFAULT_DETECTOR_PATH.exists():
        return DEFAULT_RECOGNIZER_PATH, DEFAULT_DETECTOR_PATH
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> Path:
        return Path(hf_hub_download(WEIGHTS_REPO, name, revision=WEIGHTS_REVISION))

    recognizer = fetch("best_accuracy.pth")
    fetch("opt.txt")  # the engine reads it next to the checkpoint (same snapshot folder)
    return recognizer, fetch("craft_mlt_25k.pth")


def _load() -> None:
    """Loader thread: build the engine and run one warm-up scan."""
    global _engine, _error
    try:
        import cv2 as cv

        from src.ocr_engine import OCREngine

        engine = OCREngine(*weight_paths())
        warmup = cv.imread(str(WARMUP_IMAGE))
        if warmup is not None:
            engine.process(warmup)  # so the first visitor's scan runs at full speed
        _engine = engine
    except BaseException as error:  # noqa: BLE001 - handed to whoever waits in get_engine()
        _error = error
    finally:
        _finished.set()


def start_loading() -> None:
    """Start loading the engine in the background, once per process. Cheap to call often."""
    global _loader, _error
    with _lock:
        if _engine is not None or (_loader is not None and _loader.is_alive()):
            return
        _error = None  # a failed attempt (for example a download hiccup) is retried
        _finished.clear()
        _loader = threading.Thread(target=_load, name="ocr-engine-loader", daemon=True)
        _loader.start()


def engine_ready() -> bool:
    return _engine is not None


def get_engine(timeout: float = LOAD_TIMEOUT_SECONDS) -> OCREngine:
    """The shared engine, waiting for the background load when it is still running."""
    start_loading()
    if not _finished.wait(timeout):
        raise TimeoutError("The OCR engine is still loading.")
    if _engine is None:
        raise RuntimeError("The OCR engine could not be loaded.") from _error
    return _engine


def log_scan(seconds: float, words: int, image_size: tuple[int, int]) -> None:
    """One line per scan in the server log: timing and memory only, never content (PRD N5)."""
    peak = ""
    try:
        import resource  # not available on Windows

        peak = f", peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB"
    except (ImportError, AttributeError):
        pass
    print(f"[scan] {seconds:.2f} s, {words} words, {image_size[0]}x{image_size[1]} px{peak}", flush=True)
