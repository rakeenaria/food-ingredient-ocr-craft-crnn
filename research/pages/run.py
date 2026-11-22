"""Run: read an image folder with any checkpoint and inspect every stage (PRD R1, R2)."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from research import runtime
from research.catalog import list_images, relative
from research.editing import read_image
from research.runs import ImageRun, Run, cli_overlay, line_crops, run_image, save_run
from src.ocr_engine import draw_word_boxes
from src.postprocess import compose_text, parse_ingredients
from ui import style
from ui.imaging import to_display_rgb, word_thumbnail

LOW_CONFIDENCE = 0.80
HIGHLIGHT_BGR = (77, 225, 255)  # highlighter yellow, as for low confidence in the word table
ss = st.session_state


def start_run(checkpoint, folder, *, pad: bool, refiner: bool, device: str) -> None:
    images = list_images(folder) if folder else []
    if folder is None or not folder.is_dir():
        st.error("Choose an existing folder.")
        return
    if not images:
        st.error("This folder has no PNG, JPG, BMP, TIFF or WebP images.")
        return

    config = checkpoint.config
    run = Run(
        checkpoint=checkpoint.name,
        device=device,
        settings={
            "checkpoint": checkpoint.name,
            "checkpoint_path": relative(checkpoint.path),
            "architecture": checkpoint.architecture,
            "characters": checkpoint.characters,
            "batch_max_length": str(config.batch_max_length),
            "input": f"{'RGB' if config.rgb else 'grayscale'} {config.imgH}x{config.imgW}",
            "PAD": "on" if pad else "off",
            "refiner": "on" if refiner else "off",
            "detector": "CRAFT craft_mlt_25k.pth (text 0.8, low text 0.4, link 0.25, canvas 1280, mag ratio 1.5)",
            "device": device,
            "input_folder": relative(folder),
        },
    )
    run.note(f"{checkpoint.name}: {checkpoint.architecture}, {checkpoint.characters}, PAD {'on' if pad else 'off'}")
    run.note(f"CRAFT, refiner {'on' if refiner else 'off'}, device {device}, {len(images)} images in {relative(folder)}")
    try:
        detector = runtime.load_detector(refiner, device)
        recognizer = runtime.load_recognizer(checkpoint.key, pad, device)
    except Exception as error:  # noqa: BLE001 - shown to the researcher as is
        st.error(f"The models could not be loaded: {error}")
        return

    progress = st.progress(0.0, text="Starting…")
    started = time.perf_counter()
    for number, path in enumerate(images, start=1):
        try:
            item = run_image(detector, recognizer, path)
        except Exception as error:  # noqa: BLE001
            item = ImageRun(path, None, error=f"{type(error).__name__}: {error}")
        run.images.append(item)
        if item.result is None:
            run.note(f"{path.name}: skipped ({item.error})")
        else:
            result, timings = item.result, item.result.timings
            low = sum(word.confidence < LOW_CONFIDENCE for word in result.words)
            run.note(
                f"{path.name}: {len(result.words)} words in {len(result.lines)} lines, {low} below 80%, "
                f"{timings['total']:.2f} s (detect {timings['detect']:.2f}, crop {timings['crop']:.2f}, "
                f"recognize {timings['recognize']:.2f})"
            )
        progress.progress(number / len(images), text=f"Read {number} of {len(images)}: {path.name}")
    progress.empty()
    run.note(f"Done: {len(images)} images in {time.perf_counter() - started:.1f} s")
    ss.run = run
    ss.pop("run_table", None)


def render_inspector(item: ImageRun, run: Run) -> None:
    style.section(item.path.name, relative(item.path))
    if item.result is None:
        st.warning(item.error or "This image could not be read.")
        return
    image = read_image(item.path)
    if image is None:
        st.warning("The image file can no longer be read.")
        return
    result = item.result
    overlay_tab, heatmap_tab, lines_tab, words_tab, text_tab, log_tab = st.tabs(
        ["Overlay", "Heatmaps", "Lines", "Words", "Text", "Log"]
    )
    with overlay_tab:
        modes = ["Word boxes", "Original"] + (["Refiner polygons"] if result.polygons is not None else [])
        mode = st.segmented_control("Overlay", modes, default="Word boxes", key="run_overlay", label_visibility="collapsed")
        if mode == "Original":
            shown = image
        elif mode == "Refiner polygons":
            shown = cli_overlay(image, result)
        else:
            shown = draw_word_boxes(
                image,
                result.words,
                thickness=max(2, round(max(image.shape[:2]) / 700)),
                low_confidence_threshold=LOW_CONFIDENCE,
                low_confidence_color=HIGHLIGHT_BGR,
                outline_color=(255, 255, 255),
            )
        st.image(to_display_rgb(shown), width="stretch")
        if mode in (None, "Word boxes"):
            style.note("Black boxes are the words that were read. <mark>Yellow</mark> boxes are below 80% confidence.")
        elif mode == "Refiner polygons":
            style.note("Green polygons from the CRAFT link refiner, as in the CLI overlay. The text comes from the plain word boxes.")
    with heatmap_tab:
        text_column, link_column = st.columns(2)
        with text_column:
            st.image(item.textmap_png, caption="Region score (text map)", width="stretch")
        with link_column:
            refined = run.settings.get("refiner") == "on"
            st.image(item.linkmap_png, caption=f"Affinity score (link map{', refined' if refined else ''})", width="stretch")
        style.note("CRAFT works on the image resized for its 1280 px canvas, so the maps are smaller than the photo.")
    with lines_tab:
        crops = line_crops(image, result.words)
        texts = dict(zip(sorted({word.line for word in result.words}), result.lines))
        for line_number, crop in crops:
            st.image(to_display_rgb(crop, max_side=1600), caption=f"Line {line_number}: {texts.get(line_number, '')}")
    with words_tab:
        rows = [
            (number, word_thumbnail(word.crop), word.text, word.confidence, word.confidence < LOW_CONFIDENCE)
            for number, word in enumerate(result.words, start=1)
        ]
        style.word_table(rows)
    with text_tab:
        st.code(result.text or "(no text)", language=None)
        parsed = parse_ingredients(compose_text(result.words).text)
        style.note(f"Ingredient list: {len(parsed.items)} items and {len(parsed.notes)} label notes.")
    with log_tab:
        st.code("\n".join(line for line in run.log if item.path.name in line) or "(no entries)", language=None)
        with st.expander("Whole run"):
            st.code("\n".join(run.log), language=None)


def render_results(run: Run) -> None:
    done = [item for item in run.images if item.result is not None]
    words = sum(len(item.result.words) for item in done)
    low = sum(word.confidence < LOW_CONFIDENCE for item in done for word in item.result.words)
    seconds = [item.result.timings["total"] for item in done]
    style.section("Results", f"{run.checkpoint} · {run.device.upper()}")
    style.facts(
        [
            ("Images read", f"{len(done)} of {len(run.images)}"),
            ("Words", str(words)),
            ("Words below 80% confidence", f"{low} ({low / words:.1%})" if words else "0"),
            ("Mean time per image", f"{sum(seconds) / len(seconds):.2f} s" if seconds else "–"),
        ]
    )
    table = pd.DataFrame(
        [
            {
                "Image": item.path.name,
                "Words": len(item.result.words) if item.result else 0,
                "Lines": len(item.result.lines) if item.result else 0,
                "Below 80%": sum(w.confidence < LOW_CONFIDENCE for w in item.result.words) if item.result else 0,
                "Seconds": round(item.result.timings["total"], 2) if item.result else None,
                "Status": "ok" if item.result else item.error,
            }
            for item in run.images
        ]
    )
    style.note("Select a row to inspect that image.")
    event = st.dataframe(
        table, key="run_table", on_select="rerun", selection_mode="single-row", hide_index=True, width="stretch"
    )
    rows = event.selection.rows if event and event.selection else []
    render_inspector(run.images[rows[0] if rows else 0], run)

    style.section("Save")
    style.note(
        "Writes this run in the CLI layout (crops/, craft/ with heatmaps, overlays and line crops, "
        "merged.txt and run_log.txt) to a new folder under outputs/runs/. Nothing is overwritten."
    )
    if st.button("Save run to outputs/runs/"):
        try:
            path = save_run(run)
        except OSError as error:
            st.error(f"Could not save the run: {error}")
        else:
            st.success(f"Saved to {relative(path)}")


# ---------------------------------------------------------------------- page
runtime.page_header("Run", "Read a folder of label photos with any checkpoint, then inspect every stage of the pipeline.")

checkpoints = runtime.checkpoints()
if not checkpoints:
    st.error("No recognizer checkpoints were found in saved_models/.")
    st.stop()

setup_column, options_column = st.columns([3, 2], gap="large")
with setup_column:
    style.section("Checkpoint", f"{len(checkpoints)} available")
    labels = {checkpoint.key: checkpoint.label for checkpoint in checkpoints}
    key = st.selectbox("Checkpoint", list(labels), format_func=labels.get, key="run_checkpoint", label_visibility="collapsed")
    checkpoint = next(item for item in checkpoints if item.key == key)
    config = checkpoint.config
    style.facts(
        [
            ("Architecture", checkpoint.architecture),
            ("Characters", checkpoint.characters),
            ("Max word length", str(config.batch_max_length)),
            ("Input", f"{'RGB' if config.rgb else 'Grayscale'}, {config.imgH} × {config.imgW} px"),
            ("Trained with PAD", "Yes" if config.PAD else "No"),
            ("Settings read from", checkpoint.settings_from),
        ]
    )
with options_column:
    style.section("Input and options")
    folder = runtime.folder_picker("Image folder", key="run_folder")
    pad = st.toggle(
        "Keep aspect ratio (PAD)",
        value=config.PAD,
        key=f"run_pad_{key}",
        help="Starts at the checkpoint's training setting. A different setting usually lowers accuracy.",
    )
    st.toggle(
        "RGB input",
        value=config.rgb,
        key=f"run_rgb_{key}",
        disabled=True,
        help="Fixed by the checkpoint: its first layer was trained on "
        + ("RGB" if config.rgb else "grayscale")
        + " images, so the other input cannot be loaded.",
    )
    refiner = st.toggle(
        "CRAFT link refiner",
        value=False,
        key="run_refiner",
        help="As in the CLI: adds line polygons and a refined link map for inspection. "
        "The text still comes from the plain CRAFT word boxes.",
    )
    device = runtime.device_picker("run_device")
    if st.button("Run", type="primary", width="stretch"):
        start_run(checkpoint, folder, pad=pad, refiner=refiner, device=device)

if ss.get("run"):
    render_results(ss.run)
