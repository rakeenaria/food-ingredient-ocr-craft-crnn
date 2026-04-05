import subprocess
import sys
from pathlib import Path

import cv2 as cv
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


def rotate_image(image_bgr, angle_deg: float, keep_full_frame: bool):
    """Rotate image and optionally expand canvas to keep full content."""
    height, width = image_bgr.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv.getRotationMatrix2D(center, angle_deg, 1.0)

    if keep_full_frame:
        cos_val = abs(matrix[0, 0])
        sin_val = abs(matrix[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        matrix[0, 2] += new_width / 2.0 - center[0]
        matrix[1, 2] += new_height / 2.0 - center[1]
        return cv.warpAffine(
            image_bgr,
            matrix,
            (new_width, new_height),
            flags=cv.INTER_CUBIC,
            borderMode=cv.BORDER_REPLICATE,
        )

    return cv.warpAffine(
        image_bgr,
        matrix,
        (width, height),
        flags=cv.INTER_CUBIC,
        borderMode=cv.BORDER_REPLICATE,
    )


def bgr_to_rgb(image_bgr):
    return cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)


st.set_page_config(page_title="Food Ingredient OCR", layout="wide")
st.title("Food Ingredient OCR Runner")
st.caption("Run CRAFT + TRBA OCR from UI and inspect merged output quickly.")

with st.form("ocr_form"):
    left_col, right_col = st.columns(2)

    with left_col:
        saved_model = st.text_input("Saved model path", "saved_models/FT_801010_off/best_accuracy.pth")
        input_folder = st.text_input("Input folder", "images/gambar_simulasi")
        trained_model = st.text_input("CRAFT model path", "saved_models/craft_mlt_25k.pth")

    with right_col:
        crops_folder = st.text_input("Crops folder", "outputs/crops")
        results_folder = st.text_input("Results folder", "outputs/craft")
        merged_output = st.text_input("Merged output file", "outputs/merged.txt")

    options_col_1, options_col_2 = st.columns(2)
    with options_col_1:
        refine = st.checkbox("Use CRAFT refiner", value=False)
        save_overlay = st.checkbox("Save overlay images", value=True)
        sensitive = st.checkbox("Force sensitive charset", value=True)
    with options_col_2:
        pad = st.checkbox("Use PAD", value=False)
        rgb = st.checkbox("Use RGB recognizer input", value=False)
        cuda = st.checkbox("Use CUDA", value=torch.cuda.is_available())

    run_clicked = st.form_submit_button("Run OCR")

with st.expander("Image Editor (Optional)", expanded=False):
    edit_input_folder = resolve_repo_path(input_folder.strip())
    editable_images = list_input_images(edit_input_folder)

    if not editable_images:
        st.info(f"No editable images found in: {edit_input_folder}")
    else:
        rel_paths = [str(path.relative_to(edit_input_folder)) for path in editable_images]
        selected_rel_path = st.selectbox("Select image to edit", rel_paths)
        selected_image_path = edit_input_folder / selected_rel_path
        image_bgr = cv.imread(str(selected_image_path))

        if image_bgr is None:
            st.error(f"Failed to read image: {selected_image_path}")
        else:
            angle = st.slider("Rotate angle (deg)", min_value=-30.0, max_value=30.0, value=0.0, step=0.1)
            keep_full_frame = st.checkbox("Keep full image after rotation", value=True)
            rotated = rotate_image(image_bgr, angle, keep_full_frame)

            use_crop = st.checkbox("Enable crop", value=False)
            preview = rotated
            if use_crop:
                rot_h, rot_w = rotated.shape[:2]
                crop_x = st.slider("Crop X", min_value=0, max_value=max(0, rot_w - 1), value=0, step=1)
                crop_y = st.slider("Crop Y", min_value=0, max_value=max(0, rot_h - 1), value=0, step=1)
                max_crop_w = max(1, rot_w - crop_x)
                max_crop_h = max(1, rot_h - crop_y)
                crop_w = st.slider("Crop Width", min_value=1, max_value=max_crop_w, value=max_crop_w, step=1)
                crop_h = st.slider("Crop Height", min_value=1, max_value=max_crop_h, value=max_crop_h, step=1)
                preview = rotated[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

            st.image(bgr_to_rgb(preview), caption=f"Preview: {selected_image_path.name}", use_container_width=True)

            save_col_1, save_col_2 = st.columns(2)
            with save_col_1:
                if st.button("Save (overwrite selected image)"):
                    if cv.imwrite(str(selected_image_path), preview):
                        st.success(f"Saved edited image: {selected_image_path}")
                    else:
                        st.error("Failed to save image.")
            with save_col_2:
                new_name_default = f"{selected_image_path.stem}_edited{selected_image_path.suffix}"
                new_file_name = st.text_input("Save as new filename", value=new_name_default)
                if st.button("Save as new file"):
                    output_path = selected_image_path.parent / new_file_name.strip()
                    if cv.imwrite(str(output_path), preview):
                        st.success(f"Saved edited copy: {output_path}")
                    else:
                        st.error("Failed to save edited copy.")

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

    saved_model_path = resolve_repo_path(run_options["saved_model"])
    if not saved_model_path.exists():
        st.error(f"Saved model not found: {saved_model_path}")
        st.stop()

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

    panel_height = 450
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
                for image_path in input_images:
                    st.image(str(image_path), caption=image_path.name, use_container_width=True)

    if recognized_path.exists():
        st.subheader("Recognized Table")
        recognized_df = parse_recognized_file(recognized_path)
        st.dataframe(recognized_df, use_container_width=True, height=320)
    else:
        st.warning(f"recognized.txt not found at: {recognized_path}")
