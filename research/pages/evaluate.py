"""Evaluate: score checkpoints against ground truth and compare them (PRD R3).

Two protocols, both reproducing the thesis scripts:
- Photos (end to end): test2.py on an image-level gt file.
- Word images (recognizer only): test.py on an LMDB word dataset.
Detection runs once per photo; every checkpoint reads the same word crops.
"""

from __future__ import annotations

import base64
import time

import pandas as pd
import streamlit as st

from research import runtime
from research.catalog import PROJECT_ROOT, image_folders, image_ground_truth_files, relative, word_datasets
from research.evaluation import (
    WordSample,
    detect_images,
    evaluate_photos,
    evaluate_words,
    guess_image_folder,
    match_images,
    read_image_ground_truth,
    read_lmdb,
)
from ui import style

THESIS = "Thesis protocol (test.py)"
IGNORE_CASE = "Ignore case and symbols"
ss = st.session_state


@st.cache_data(show_spinner="Reading the word dataset…", max_entries=2)
def load_word_samples(path: str) -> list[WordSample]:
    return read_lmdb(PROJECT_ROOT / path)


def image_data_url(data: bytes) -> str:
    mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def number_columns(columns: dict[str, str]) -> dict:
    return {name: st.column_config.NumberColumn(name, format=fmt) for name, fmt in columns.items()}


# ------------------------------------------------------------------ photos
def run_photo_evaluation(selected, labels, configs, matched, *, device: str, gt: str, folder: str) -> None:
    progress = st.progress(0.0, text="Loading CRAFT…")
    try:
        detector = runtime.load_detector(False, device)
        detections = detect_images(
            detector,
            matched,
            on_progress=lambda n, total: progress.progress(0.3 * n / total, text=f"Finding words: photo {n} of {total}"),
        )
        reports = []
        for number, key in enumerate(selected):
            progress.progress(0.3 + 0.7 * number / len(selected), text=f"Reading with {labels[key]}…")
            started = time.perf_counter()
            recognizer = runtime.load_recognizer(key, configs[key].PAD, device)
            reports.append((key, evaluate_photos(recognizer, detections), time.perf_counter() - started))
    except Exception as error:  # noqa: BLE001 - shown to the researcher as is
        progress.empty()
        st.error(f"The evaluation stopped: {type(error).__name__}: {error}")
        return
    progress.empty()
    runtime.release_gpu_memory()
    ss.eval_photos = {"gt": gt, "folder": folder, "device": device, "labels": dict(labels), "reports": reports}


def render_photo_results() -> None:
    data = ss.get("eval_photos")
    if not data:
        return
    reports = sorted(data["reports"], key=lambda item: item[1].word_accuracy, reverse=True)
    labels = data["labels"]
    style.section("Word accuracy", f"{data['gt']} · {reports[0][1].samples} photos")
    style.meters(
        (
            labels[key],
            report.word_accuracy,
            f"CER {report.cer:.3f} · WER {report.wer:.3f} · F1 {report.f1:.3f}",
        )
        for key, report, _ in reports
    )
    style.note("Word accuracy is 1 − WER over all photos, as in test2.py. Hover a row for more.")
    table = pd.DataFrame(
        [
            {
                "Checkpoint": labels[key],
                "Word accuracy (%)": report.word_accuracy,
                "CER": report.cer,
                "WER": report.wer,
                "norm_ED": report.norm_ed,
                "F1": report.f1,
                "BLEU-1": report.bleu1,
                "ROUGE-L": report.rouge_l,
                "GT words": report.gt_words,
                "Predicted words": report.pred_words,
                "Seconds": seconds,
            }
            for key, report, seconds in reports
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config=number_columns(
            {"Word accuracy (%)": "%.3f", "CER": "%.3f", "WER": "%.3f", "norm_ED": "%.3f", "F1": "%.3f",
             "BLEU-1": "%.3f", "ROUGE-L": "%.3f", "Seconds": "%.1f"}
        ),
    )

    style.section("Per photo")
    keys = [key for key, _, _ in reports]
    key = st.selectbox("Checkpoint", keys, format_func=labels.get, key="eval_photo_detail")
    report = next(report for item_key, report, _ in reports if item_key == key)
    per_image = pd.DataFrame(
        [
            {
                "Photo": score.key,
                "Word accuracy (%)": score.word_accuracy,
                "CER": score.cer,
                "WER": score.wer,
                "norm_ED": score.norm_ed,
                "F1": score.f1,
                "GT words": score.gt_words,
                "Predicted words": score.pred_words,
            }
            for score in report.images
        ]
    )
    event = st.dataframe(
        per_image,
        key=f"eval_photo_table_{key}",
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        width="stretch",
        column_config=number_columns({"Word accuracy (%)": "%.3f", "CER": "%.3f", "WER": "%.3f", "norm_ED": "%.3f", "F1": "%.3f"}),
    )
    rows = event.selection.rows if event and event.selection else []
    score = report.images[rows[0] if rows else 0]
    style.note(f"Select a row to compare texts. Showing <b>{score.key}</b>; words that differ are highlighted.")
    style.text_diff(score.ground_truth, score.prediction, reference_label="Ground truth", candidate_label="Prediction")


def render_photos_tab(selected, labels, configs, device: str) -> None:
    files = image_ground_truth_files()
    if not files:
        st.info("No image-level gt*.txt files were found in data/.")
        return
    options = {relative(item.path): f"{relative(item.path)}  ({item.entries} photos)" for item in files}
    left, right = st.columns(2, gap="large")
    with left:
        gt = st.selectbox("Ground truth", list(options), format_func=options.get, key="eval_gt")
    truths = read_image_ground_truth(PROJECT_ROOT / gt)
    folders = [path for path, _ in image_folders()]
    guess = guess_image_folder(truths, folders)
    counts = {relative(path): len(match_images(truths, path)[0]) for path in folders}
    names = list(counts)
    with right:
        folder = st.selectbox(
            "Photos",
            names,
            index=names.index(relative(guess)) if guess else 0,
            format_func=lambda name: f"{name}  ({counts[name]} of {len(truths)} matched)",
            key=f"eval_folder_{gt}",
        )
    matched, missing = match_images(truths, PROJECT_ROOT / folder)
    if missing:
        listed = ", ".join(truth.name for truth in missing[:6]) + (" …" if len(missing) > 6 else "")
        style.note(f"{len(missing)} entries have no photo in this folder and are skipped: {listed}")
    if st.button("Evaluate photos", type="primary", disabled=not selected or not matched):
        run_photo_evaluation(selected, labels, configs, matched, device=device, gt=gt, folder=folder)
    render_photo_results()


# ------------------------------------------------------------------- words
def run_word_evaluation(selected, labels, configs, *, dataset: str, protocol: str, device: str) -> None:
    samples = load_word_samples(dataset)
    progress = st.progress(0.0, text="Starting…")
    reports = []
    try:
        for number, key in enumerate(selected):
            progress.progress(number / len(selected), text=f"Reading {len(samples)} words with {labels[key]}…")
            started = time.perf_counter()
            recognizer = runtime.load_recognizer(key, configs[key].PAD, device)
            report = evaluate_words(recognizer, samples, filtering_off=protocol == IGNORE_CASE)
            reports.append((key, report, time.perf_counter() - started))
    except Exception as error:  # noqa: BLE001
        progress.empty()
        st.error(f"The evaluation stopped: {type(error).__name__}: {error}")
        return
    progress.empty()
    runtime.release_gpu_memory()
    ss.eval_words = {"dataset": dataset, "protocol": protocol, "labels": dict(labels), "reports": reports, "total": len(samples)}


def render_word_results() -> None:
    data = ss.get("eval_words")
    if not data:
        return
    reports = sorted(data["reports"], key=lambda item: item[1].accuracy, reverse=True)
    labels = data["labels"]
    style.section("Accuracy", f"{data['dataset']} · {data['protocol']}")
    style.meters(
        (
            labels[key],
            report.accuracy,
            f"{report.samples} words · CER {report.cer:.3f} · WER {report.wer:.3f} · norm_ED {report.norm_ed:.3f}",
        )
        for key, report, _ in reports
    )
    if data["protocol"] == THESIS:
        style.note(
            "Exact match, as in test.py. Like the thesis data loader, each checkpoint skips labels that are longer "
            "than its maximum length or use characters it cannot output, so the word count can differ."
        )
    else:
        style.note("Upper and lower case count as the same and only letters and digits are compared, on every word.")
    table = pd.DataFrame(
        [
            {
                "Checkpoint": labels[key],
                "Words": report.samples,
                "Accuracy (%)": report.accuracy,
                "norm_ED": report.norm_ed,
                "CER": report.cer,
                "WER": report.wer,
                "F1": report.f1,
                "Avg confidence": report.avg_confidence,
                "Coverage@0.8": report.coverage,
                "Accuracy@0.8": report.accuracy_at_threshold,
                "Seconds": seconds,
            }
            for key, report, seconds in reports
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config=number_columns(
            {"Accuracy (%)": "%.3f", "norm_ED": "%.3f", "CER": "%.3f", "WER": "%.3f", "F1": "%.3f",
             "Avg confidence": "%.3f", "Coverage@0.8": "%.3f", "Accuracy@0.8": "%.3f", "Seconds": "%.1f"}
        ),
    )

    style.section("Mistakes")
    keys = [key for key, _, _ in reports]
    key = st.selectbox("Checkpoint", keys, format_func=labels.get, key="eval_word_detail")
    report = next(report for item_key, report, _ in reports if item_key == key)
    samples = load_word_samples(data["dataset"])
    wrong = [row for row in report.rows if not row[4]]
    if not wrong:
        st.success("No mistakes.")
        return
    style.note(f"{len(wrong)} of {report.samples} words are wrong, lowest confidence first.")
    mistakes = pd.DataFrame(
        [
            {
                "Image": image_data_url(samples[index].image),
                "Ground truth": label,
                "Prediction": prediction,
                "Confidence": confidence,
            }
            for index, label, prediction, confidence, _ in sorted(wrong, key=lambda row: row[3])
        ]
    )
    st.dataframe(
        mistakes,
        hide_index=True,
        width="stretch",
        row_height=44,
        column_config={
            "Image": st.column_config.ImageColumn("Image", width="medium"),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
        },
    )


def render_words_tab(selected, labels, configs, device: str) -> None:
    datasets = word_datasets()
    if not datasets:
        st.info("No LMDB word datasets (folders with data.mdb) were found in data/.")
        return
    options = {relative(item.path): f"{relative(item.path)}  ({item.samples} words)" for item in datasets}
    left, right = st.columns(2, gap="large")
    with left:
        dataset = st.selectbox("Word dataset", list(options), format_func=options.get, key="eval_dataset")
    with right:
        protocol = st.radio(
            "Protocol",
            [THESIS, IGNORE_CASE],
            key="eval_protocol",
            horizontal=True,
            help="The thesis protocol reproduces test.py. The other one is test.py with --data_filtering_off "
            "on a case-sensitive model: fair for comparing models with different character sets.",
        )
    if st.button("Evaluate word images", type="primary", disabled=not selected):
        run_word_evaluation(selected, labels, configs, dataset=dataset, protocol=protocol, device=device)
    render_word_results()


# ---------------------------------------------------------------------- page
runtime.page_header("Evaluate", "Score checkpoints against ground truth with the thesis protocols, side by side.")

checkpoints = runtime.checkpoints()
if not checkpoints:
    st.error("No recognizer checkpoints were found in saved_models/.")
    st.stop()
labels = {checkpoint.key: checkpoint.label for checkpoint in checkpoints}
configs = {checkpoint.key: checkpoint.config for checkpoint in checkpoints}

choose_column, device_column = st.columns([3, 1], gap="large")
with choose_column:
    style.section("Checkpoints to compare")
    selected = st.multiselect(
        "Checkpoints",
        list(labels),
        default=[checkpoints[0].key],
        format_func=labels.get,
        key="eval_checkpoints",
        label_visibility="collapsed",
        placeholder="Choose one or more checkpoints",
    )
with device_column:
    style.section("Device")
    device = runtime.device_picker("eval_device", label_visibility="collapsed")

photos_tab, words_tab = st.tabs(["Photos, end to end (test2.py)", "Word images, recognizer only (test.py)"])
with photos_tab:
    render_photos_tab(selected, labels, configs, device)
with words_tab:
    render_words_tab(selected, labels, configs, device)
