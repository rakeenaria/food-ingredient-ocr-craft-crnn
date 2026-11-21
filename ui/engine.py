"""Process-wide OCR engine for the Streamlit app (loaded once, shared by sessions).

Locally the weights come from saved_models/. They are not in git, so on the
hosting platform they are downloaded once from the public Hugging Face model
repo (no token needed) and kept in the Hugging Face cache. See docs/DEPLOY.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2 as cv
import streamlit as st

from src.ocr_engine import DEFAULT_DETECTOR_PATH, DEFAULT_RECOGNIZER_PATH, PROJECT_ROOT, OCREngine

WARMUP_IMAGE = PROJECT_ROOT / "images" / "gambar_simulasi" / "04.jpeg"
# Public model repo holding best_accuracy.pth, opt.txt and craft_mlt_25k.pth.
WEIGHTS_REPO = os.environ.get("INGREDIENT_READER_WEIGHTS_REPO", "rakeenaria/ingredient-reader")
WEIGHTS_REVISION = os.environ.get("INGREDIENT_READER_WEIGHTS_REVISION", "main")


def weight_paths(*, prefer_local: bool = True) -> tuple[Path, Path]:
    """(recognizer, detector): the local files when present, otherwise the Hub copies."""
    if prefer_local and DEFAULT_RECOGNIZER_PATH.exists() and DEFAULT_DETECTOR_PATH.exists():
        return DEFAULT_RECOGNIZER_PATH, DEFAULT_DETECTOR_PATH
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> Path:
        return Path(hf_hub_download(WEIGHTS_REPO, name, revision=WEIGHTS_REVISION))

    recognizer = fetch("best_accuracy.pth")
    fetch("opt.txt")  # the engine reads it next to the checkpoint (same snapshot folder)
    return recognizer, fetch("craft_mlt_25k.pth")


@st.cache_resource(show_spinner=False)
def get_engine() -> OCREngine:
    """Load both models once, then run one warm-up scan so the first visitor is not slowed down."""
    recognizer_path, detector_path = weight_paths()
    engine = OCREngine(recognizer_path, detector_path)
    warmup = cv.imread(str(WARMUP_IMAGE))
    if warmup is not None:
        engine.process(warmup)
    return engine


def log_scan(seconds: float, words: int, image_size: tuple[int, int]) -> None:
    """One line per scan in the server log: timing and memory only, never content (PRD N5)."""
    peak = ""
    try:
        import resource  # not available on Windows

        peak = f", peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB"
    except (ImportError, AttributeError):
        pass
    print(f"[scan] {seconds:.2f} s, {words} words, {image_size[0]}x{image_size[1]} px{peak}", flush=True)
