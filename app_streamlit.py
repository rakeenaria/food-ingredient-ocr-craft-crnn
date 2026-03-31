import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_repo_path(raw_value: str) -> Path:
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_demo_command(options: dict) -> list[str]:
    command = [
        sys.executable,
        "demo.py",
        "--saved_model",
        options["saved_model"],
        "--input_folder",
        options["input_folder"],
        "--trained_model",
        options["trained_model"],
        "--crops_folder",
        options["crops_folder"],
        "--results_folder",
        options["results_folder"],
        "--merged_output",
        options["merged_output"],
    ]

    if options["refine"]:
        command.append("--refine")
    if options["save_overlay"]:
        command.append("--save_overlay")
    if options["sensitive"]:
        command.append("--sensitive")
    if options["pad"]:
        command.append("--PAD")
    if options["rgb"]:
        command.append("--rgb")
    if options["cuda"]:
        command.extend(["--cuda", "true"])
    else:
        command.extend(["--cuda", "false"])
    return command


def parse_recognized_file(recognized_path: Path) -> pd.DataFrame:
    rows = []
    for line in recognized_path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            crop_path, text = line.split("\t", 1)
        else:
            crop_path, text = line, ""
        rows.append({"crop_path": crop_path, "text": text})
    return pd.DataFrame(rows)


def list_input_images(input_folder: Path) -> list[Path]:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not input_folder.exists():
        return []
    images = [path for path in input_folder.rglob("*") if path.is_file() and path.suffix.lower() in image_exts]
    images.sort()
    return images


st.set_page_config(page_title="Food Ingredient OCR", layout="wide")
st.title("Food Ingredient OCR Runner")
st.caption("Run CRAFT + TRBA OCR from UI and inspect merged output quickly.")

with st.form("ocr_form"):
    left_col, right_col = st.columns(2)

    with left_col:
        saved_model = st.text_input("Saved model path", "fine_tuned_model/best_accuracy.pth")
        input_folder = st.text_input("Input folder", "bahan")
        trained_model = st.text_input("CRAFT model path", "saved_models/craft_mlt_25k.pth")

    with right_col:
        crops_folder = st.text_input("Crops folder", "outputs/crops")
        results_folder = st.text_input("Results folder", "outputs/craft")
        merged_output = st.text_input("Merged output file", "outputs/merged.txt")

    options_col_1, options_col_2 = st.columns(2)
    with options_col_1:
        refine = st.checkbox("Use CRAFT refiner", value=True)
        save_overlay = st.checkbox("Save overlay images", value=True)
        sensitive = st.checkbox("Force sensitive charset", value=True)
    with options_col_2:
        pad = st.checkbox("Use PAD", value=False)
        rgb = st.checkbox("Use RGB recognizer input", value=False)
        cuda = st.checkbox("Use CUDA", value=torch.cuda.is_available())

    run_clicked = st.form_submit_button("Run OCR")

if run_clicked:
    run_options = {
        "saved_model": saved_model.strip(),
        "input_folder": input_folder.strip(),
        "trained_model": trained_model.strip(),
        "crops_folder": crops_folder.strip(),
        "results_folder": results_folder.strip(),
        "merged_output": merged_output.strip(),
        "refine": refine,
        "save_overlay": save_overlay,
        "sensitive": sensitive,
        "pad": pad,
        "rgb": rgb,
        "cuda": cuda,
    }

    command = build_demo_command(run_options)
    st.code(" ".join(command), language="bash")

    with st.spinner("Running OCR pipeline..."):
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    if completed.returncode != 0:
        st.error(f"Demo failed (exit code {completed.returncode})")
    else:
        st.success("Demo finished successfully.")

    if completed.stdout:
        st.subheader("Stdout")
        st.text_area("stdout", completed.stdout, height=280)
    if completed.stderr:
        st.subheader("Stderr")
        st.text_area("stderr", completed.stderr, height=180)

    merged_path = resolve_repo_path(run_options["merged_output"])
    recognized_path = resolve_repo_path(run_options["results_folder"]) / "recognized.txt"
    input_folder_path = resolve_repo_path(run_options["input_folder"])
    input_images = list_input_images(input_folder_path)

    panel_height = 420
    merged_col, images_col = st.columns(2)
    with merged_col:
        st.subheader("Merged Output")
        if merged_path.exists():
            st.text_area("merged.txt", merged_path.read_text(encoding="utf-8"), height=panel_height)
        else:
            st.warning(f"Merged output not found at: {merged_path}")

    with images_col:
        st.subheader("Input Images")
        with st.container(height=panel_height, border=True):
            if not input_images:
                st.warning(f"No images found in: {input_folder_path}")
            else:
                grid_cols = st.columns(2)
                for index, image_path in enumerate(input_images):
                    with grid_cols[index % 2]:
                        st.image(str(image_path), caption=image_path.name, use_container_width=True)

    if recognized_path.exists():
        st.subheader("Recognized Table")
        recognized_df = parse_recognized_file(recognized_path)
        st.dataframe(recognized_df, use_container_width=True, height=320)
    else:
        st.warning(f"recognized.txt not found at: {recognized_path}")
