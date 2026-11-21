"""Process-wide OCR engine for the Streamlit app (loaded once, shared by sessions)."""

from __future__ import annotations

import cv2 as cv
import streamlit as st

from src.ocr_engine import PROJECT_ROOT, OCREngine

WARMUP_IMAGE = PROJECT_ROOT / "images" / "gambar_simulasi" / "04.jpeg"


@st.cache_resource(show_spinner=False)
def get_engine() -> OCREngine:
    """Load both models once, then run one warm-up scan so the first visitor is not slowed down."""
    engine = OCREngine()
    warmup = cv.imread(str(WARMUP_IMAGE))
    if warmup is not None:
        engine.process(warmup)
    return engine
