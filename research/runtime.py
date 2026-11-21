"""Streamlit helpers shared by the researcher pages: cached models and pickers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import streamlit as st
import torch

from research.catalog import (
    DETECTOR_PATH,
    PROJECT_ROOT,
    REFINER_PATH,
    Checkpoint,
    discover_checkpoints,
    image_folders,
    relative,
)
from src.ocr_engine import TextDetector, WordRecognizer
from ui import style

OTHER_FOLDER = "Other folder…"
DEFAULT_FOLDER = "images/gambar_simulasi"  # the five thesis photos


def page_header(title: str, lede: str) -> None:
    style.inject(wide=True)
    style.masthead(title, lede)


def checkpoints() -> list[Checkpoint]:
    """Read fresh on every run, so new checkpoints appear without a restart."""
    return discover_checkpoints()


def checkpoint_by_key(key: str) -> Checkpoint | None:
    return next((checkpoint for checkpoint in discover_checkpoints() if checkpoint.key == key), None)


def device_picker(key: str, *, label_visibility: str = "visible") -> str:
    """Auto / GPU / CPU, resolved to "cuda" or "cpu"."""
    has_cuda = torch.cuda.is_available()
    options = ["Auto", "GPU", "CPU"] if has_cuda else ["Auto", "CPU"]
    choice = st.segmented_control("Device", options, default="Auto", key=key, label_visibility=label_visibility) or "Auto"
    if choice == "Auto":
        return "cuda" if has_cuda else "cpu"
    return "cuda" if choice == "GPU" else "cpu"


@st.cache_resource(max_entries=1, show_spinner="Loading the CRAFT detector…")
def load_detector(refiner: bool, device: str) -> TextDetector:
    return TextDetector(DETECTOR_PATH, device=device, refiner_path=REFINER_PATH if refiner else None)


@st.cache_resource(max_entries=2, show_spinner="Loading the checkpoint…")
def load_recognizer(checkpoint_key: str, pad: bool, device: str) -> WordRecognizer:
    checkpoint = checkpoint_by_key(checkpoint_key)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_key}")
    return WordRecognizer(checkpoint.path, config=replace(checkpoint.config, PAD=pad), device=device)


def release_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def folder_picker(label: str, key: str) -> Path | None:
    """A known image folder, or any folder typed by path."""
    folders = image_folders()
    counts = {relative(path): count for path, count in folders}
    options = [*counts, OTHER_FOLDER]
    choice = st.selectbox(
        label,
        options,
        index=options.index(DEFAULT_FOLDER) if DEFAULT_FOLDER in options else 0,
        format_func=lambda option: f"{option}  ({counts[option]} images)" if option in counts else option,
        key=key,
    )
    if choice != OTHER_FOLDER:
        return PROJECT_ROOT / choice
    typed = st.text_input(
        "Folder path",
        key=f"{key}_path",
        placeholder="images/my_labels or C:/Users/me/Pictures/labels",
    ).strip()
    if not typed:
        return None
    path = Path(typed).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
