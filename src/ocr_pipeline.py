import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# Detection components
from CRAFT.craft import CRAFT
import CRAFT.craft_utils as craft_utils
import imgproc

# Recognition components
from dataset import RawDataset, AlignCollate
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return points in a stable order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    point_sums = points.sum(axis=1)
    rect[0] = points[np.argmin(point_sums)]
    rect[2] = points[np.argmax(point_sums)]

    point_diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(point_diffs)]
    rect[3] = points[np.argmax(point_diffs)]
    return rect


def perspective_crop_from_quad(image_bgr: np.ndarray, box: np.ndarray):
    """Crop text using perspective transform so rotated text remains readable."""
    points = np.array(box, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 4:
        return None

    ordered_points = order_quad_points(points[:4])
    top_left, top_right, bottom_right, bottom_left = ordered_points

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    target_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    target_height = int(max(height_a, height_b))

    if target_width < 2 or target_height < 2:
        return None

    destination = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype="float32",
    )
    transform_matrix = cv.getPerspectiveTransform(ordered_points, destination)
    return cv.warpPerspective(image_bgr, transform_matrix, (target_width, target_height), flags=cv.INTER_CUBIC)


def compute_axis_aligned_iou(quad_a, quad_b) -> float:
    """Compute IoU using axis-aligned bounds from two quadrilateral boxes."""
    box_a = np.array(quad_a)
    box_b = np.array(quad_b)

    a_x0, a_y0 = box_a[:, 0].min(), box_a[:, 1].min()
    a_x1, a_y1 = box_a[:, 0].max(), box_a[:, 1].max()
    b_x0, b_y0 = box_b[:, 0].min(), box_b[:, 1].min()
    b_x1, b_y1 = box_b[:, 0].max(), box_b[:, 1].max()

    inter_x0 = max(a_x0, b_x0)
    inter_y0 = max(a_y0, b_y0)
    inter_x1 = min(a_x1, b_x1)
    inter_y1 = min(a_y1, b_y1)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = (a_x1 - a_x0) * (a_y1 - a_y0)
    area_b = (b_x1 - b_x0) * (b_y1 - b_y0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def sort_boxes_reading_order(boxes, min_tol: float = 10.0):
    """Sort boxes top-to-bottom then left-to-right using baseline-based line grouping."""
    if len(boxes) == 0:
        return []

    stats = []  # (index, min_x, baseline_y, height)
    for index, box in enumerate(boxes):
        quad = np.array(box)
        ys = quad[:, 1]
        xs = quad[:, 0]
        min_x = float(xs.min())
        min_y, max_y = float(ys.min()), float(ys.max())
        height = max_y - min_y
        ys_sorted = np.sort(ys)
        baseline_y = float(np.mean(ys_sorted[-2:])) if ys_sorted.shape[0] >= 2 else float(max_y)
        stats.append((index, min_x, baseline_y, height))

    median_height = float(np.median([item[3] for item in stats])) if stats else 0.0
    line_tolerance = max(min_tol, 0.5 * median_height) if median_height > 0 else min_tol

    stats.sort(key=lambda item: item[2])
    grouped_lines = []
    current_line = []
    running_baseline = None

    for item in stats:
        if current_line and running_baseline is not None and abs(item[2] - running_baseline) > line_tolerance:
            grouped_lines.append(current_line)
            current_line = [item]
            running_baseline = item[2]
        else:
            current_line.append(item)
            if running_baseline is None:
                running_baseline = item[2]
            else:
                running_baseline = (running_baseline * (len(current_line) - 1) + item[2]) / len(current_line)

    if current_line:
        grouped_lines.append(current_line)

    order = []
    for line in grouped_lines:
        line.sort(key=lambda item: item[1])
        order.extend([item[0] for item in line])
    return order


def strip_module_prefix(state_dict):
    """Remove DataParallel prefix `module.` when present."""
    if len(state_dict) > 0 and list(state_dict.keys())[0].startswith("module."):
        normalized = {}
        for key, value in state_dict.items():
            normalized[key[len("module.") :]] = value
        return normalized
    return state_dict


def load_craft_detector(weight_path: str, use_cuda: bool):
    detector = CRAFT()
    if use_cuda:
        detector.load_state_dict(strip_module_prefix(torch.load(weight_path)))
        detector = detector.cuda()
        detector = torch.nn.DataParallel(detector)
        cudnn.benchmark = False
    else:
        detector.load_state_dict(strip_module_prefix(torch.load(weight_path, map_location="cpu")))
    detector.eval()
    return detector


def detect_text_regions(detector, image_bgr, args, refine_net=None):
    """Run CRAFT detection and return boxes, polygons, and score maps."""
    resized_image, target_ratio, _ = imgproc.resize_aspect_ratio(
        image_bgr,
        args.canvas_size,
        interpolation=cv.INTER_LINEAR,
        mag_ratio=args.mag_ratio,
    )
    ratio_h = ratio_w = 1 / target_ratio

    normalized = imgproc.normalizeMeanVariance(resized_image)
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)
    tensor = Variable(tensor.unsqueeze(0))
    if args.cuda:
        tensor = tensor.cuda()

    with torch.no_grad():
        prediction, feature_map = detector(tensor)

    score_text = prediction[0, :, :, 0].cpu().data.numpy()
    score_link = prediction[0, :, :, 1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            refined = refine_net(prediction, feature_map)
        score_link = refined[0, :, :, 0].cpu().data.numpy()

    boxes, polygons = craft_utils.getDetBoxes(
        score_text,
        score_link,
        args.text_threshold,
        args.link_threshold,
        args.low_text,
        args.poly,
    )
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    try:
        polygons = craft_utils.adjustResultCoordinates(polygons, ratio_w, ratio_h)
    except Exception:
        polygons = boxes

    for idx in range(len(polygons)):
        if polygons[idx] is None:
            polygons[idx] = boxes[idx]

    return boxes, polygons, score_text, score_link


def build_recognizer_model(opt):
    """Build TRBA recognizer and load checkpoint."""
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)

    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3

    recognizer = Model(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "Recognizer params",
        opt.imgH,
        opt.imgW,
        opt.num_fiducial,
        opt.input_channel,
        opt.output_channel,
        opt.hidden_size,
        opt.num_class,
        opt.batch_max_length,
        opt.Transformation,
        opt.FeatureExtraction,
        opt.SequenceModeling,
        opt.Prediction,
    )

    print(f"Loading recognizer weights from {opt.saved_model}")
    state = torch.load(opt.saved_model, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        recognizer.load_state_dict(state)
    except Exception:
        from collections import OrderedDict

        stripped = OrderedDict((k[7:], v) if k.startswith("module.") else (k, v) for k, v in state.items())
        recognizer.load_state_dict(stripped, strict=False)

    recognizer = torch.nn.DataParallel(recognizer).to(device)
    recognizer.eval()
    return recognizer, converter, device


def recognize_crop_folder(model, converter, device, crops_dir: Path, opt):
    """Run batched recognition on cropped word images."""
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    dataset = RawDataset(root=str(crops_dir), opt=opt)
    if len(dataset) == 0:
        return []

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=align_collate,
        pin_memory=True,
    )

    results = []
    with torch.no_grad():
        for image_tensors, image_paths in data_loader:
            batch_size = image_tensors.size(0)
            image_batch = image_tensors.to(device)
            max_lengths = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if "CTC" in opt.Prediction:
                predictions = model(image_batch, text_for_pred)
                prediction_sizes = torch.IntTensor([predictions.size(1)] * batch_size)
                _, prediction_indices = predictions.max(2)
                prediction_texts = converter.decode(prediction_indices, prediction_sizes)
            else:
                predictions = model(image_batch, text_for_pred, is_train=False)
                _, prediction_indices = predictions.max(2)
                prediction_texts = converter.decode(prediction_indices, max_lengths)
                prediction_texts = [text.split("[s]")[0] for text in prediction_texts]

            for path, text in zip(image_paths, prediction_texts):
                results.append((path, text))

    return results


def parse_word_order_index(stem_name: str):
    """Parse sort key from crop file naming convention for reconstruction."""
    if "_line" in stem_name and "_word" in stem_name:
        base_name, tail = stem_name.split("_line", 1)
        line_no = 0
        word_no = 0
        try:
            parts = tail.split("_word")
            line_no = int(parts[0])
            if len(parts) > 1:
                word_no = int(parts[1])
        except Exception:
            pass
        return base_name, line_no * 1000 + word_no

    if "_box_" in stem_name:
        base_name, index_text = stem_name.rsplit("_box_", 1)
        try:
            return base_name, int(index_text)
        except Exception:
            return base_name, 0

    return stem_name, 0


def write_merged_outputs(recognition_results, results_dir: Path):
    """Reconstruct paragraph-level text in reading order and write output files."""
    grouped_predictions = defaultdict(list)

    for image_path, predicted_text in recognition_results:
        stem_name = Path(image_path).stem
        # Skip line-only crops to avoid duplicate text in final merge.
        if "_line" in stem_name and "_word" not in stem_name:
            continue

        base_name, order_index = parse_word_order_index(stem_name)
        grouped_predictions[base_name].append((order_index, predicted_text))

    merged_output_path = results_dir / "merged.txt"
    with open(merged_output_path, "w", encoding="utf-8") as merged_file:
        for base_name, items in grouped_predictions.items():
            items.sort(key=lambda item: item[0])

            lines = defaultdict(list)
            for order_index, text in items:
                line_no = order_index // 1000
                lines[line_no].append((order_index, text))

            ordered_lines = []
            for line_no in sorted(lines.keys()):
                words = lines[line_no]
                words.sort(key=lambda item: item[0])
                line_text = " ".join(word for _, word in words)
                ordered_lines.append(line_text)

                line_output_path = results_dir / f"res_{base_name}_line{line_no}_merged.txt"
                with open(line_output_path, "w", encoding="utf-8") as line_file:
                    line_file.write(line_text + "\n")

            paragraph = "\n".join(ordered_lines)
            image_output_path = results_dir / f"res_{base_name}_merged.txt"
            with open(image_output_path, "w", encoding="utf-8") as image_file:
                image_file.write("\n".join(ordered_lines) + "\n")

            merged_file.write(f"{base_name}\t{paragraph}\n\n")
            print(f"\n[{base_name}]: {paragraph}")

    print(f"Merged paragraphs saved to {merged_output_path}")


def build_argument_parser():
    parser = argparse.ArgumentParser()

    # Detection arguments
    parser.add_argument("--input_folder", default="./bahan", help="input image folder (recursive)")
    parser.add_argument("--trained_model", default="saved_models/craft_mlt_25k.pth", help="CRAFT weight path")
    parser.add_argument("--text_threshold", type=float, default=0.7)
    parser.add_argument("--low_text", type=float, default=0.4)
    parser.add_argument("--link_threshold", type=float, default=0.4)
    parser.add_argument("--canvas_size", type=int, default=1280)
    parser.add_argument("--mag_ratio", type=float, default=1.5)
    parser.add_argument("--poly", action="store_true")
    parser.add_argument("--refine", action="store_true", help="enable CRAFT link refiner")
    parser.add_argument(
        "--refiner_model",
        default="saved_models/craft_refiner_CTW1500.pth",
        help="CRAFT refiner weight path",
    )
    parser.add_argument(
        "--cuda",
        type=lambda value: str(value).lower() in ("1", "true", "yes", "y"),
        default=torch.cuda.is_available(),
    )
    parser.add_argument("--crops_folder", default="./outputs/crops", help="folder to save cropped text regions")
    parser.add_argument("--results_folder", default="./outputs/craft", help="folder to save output text and maps")
    parser.add_argument("--save_overlay", action="store_true", help="save visualized detection overlay")

    # Recognition arguments
    parser.add_argument("--saved_model", required=True, help="recognition checkpoint path (.pth)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_max_length", type=int, default=25)
    parser.add_argument("--imgH", type=int, default=32)
    parser.add_argument("--imgW", type=int, default=100)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--character", type=str, default="0123456789abcdefghijklmnopqrstuvwxyz")
    parser.add_argument("--sensitive", action="store_true")
    parser.add_argument("--PAD", action="store_true")
    parser.add_argument("--Transformation", type=str, default="TPS")
    parser.add_argument("--FeatureExtraction", type=str, default="ResNet")
    parser.add_argument("--SequenceModeling", type=str, default="BiLSTM")
    parser.add_argument("--Prediction", type=str, default="Attn")
    parser.add_argument("--num_fiducial", type=int, default=20)
    parser.add_argument("--input_channel", type=int, default=1)
    parser.add_argument("--output_channel", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=256)
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.sensitive:
        import string

        args.character = string.printable[:-6]

    input_dir = Path(args.input_folder)
    crops_dir = Path(args.crops_folder)
    results_dir = Path(args.results_folder)

    # Always rebuild crop output so each run has deterministic contents.
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    line_crops_dir = crops_dir / "lines"
    line_crops_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    recognizer, label_converter, device = build_recognizer_model(args)
    craft_detector = load_craft_detector(args.trained_model, args.cuda)

    refiner = None
    if args.refine:
        try:
            from refinenet import RefineNet

            refiner = RefineNet()
            if args.cuda:
                refiner.load_state_dict(strip_module_prefix(torch.load(args.refiner_model)))
                refiner = refiner.cuda()
                refiner = torch.nn.DataParallel(refiner)
            else:
                refiner.load_state_dict(strip_module_prefix(torch.load(args.refiner_model, map_location="cpu")))
            refiner.eval()
            args.poly = True
            print(f"Loaded refiner from {args.refiner_model}")
        except Exception as error:
            print(f"Warning: failed to load refiner: {error}")
            refiner = None

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    input_images = [path for path in input_dir.rglob("*") if path.suffix.lower() in image_extensions]
    print(f"Found {len(input_images)} images in {input_dir}")

    total_crops_written = 0

    for image_path in input_images:
        image_bgr = cv.imread(str(image_path))
        if image_bgr is None:
            print(f"Skip (cannot read): {image_path}")
            continue

        # First pass can use refiner for better grouping quality.
        boxes_refined, polygons_refined, score_text, score_link = detect_text_regions(
            craft_detector,
            image_bgr,
            args,
            refine_net=refiner,
        )

        image_stem = image_path.stem
        cv.imwrite(str(results_dir / f"{image_stem}_textmap.png"), imgproc.cvt2HeatmapImg(score_text))
        cv.imwrite(str(results_dir / f"{image_stem}_linkmap.png"), imgproc.cvt2HeatmapImg(score_link))

        if boxes_refined is None or len(boxes_refined) == 0:
            print(f"{image_path} -> 0 boxes")
            continue

        # Plain CRAFT pass is used to stabilize reading-order sorting.
        boxes_plain, _, _, _ = detect_text_regions(craft_detector, image_bgr, args, refine_net=None)
        if boxes_plain is None or len(boxes_plain) == 0:
            boxes_plain = boxes_refined

        if args.save_overlay:
            overlay = image_bgr.copy()
            polygons_for_draw = polygons_refined if len(polygons_refined) == len(boxes_refined) else boxes_refined
            for polygon in polygons_for_draw:
                points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv.polylines(overlay, [points], True, (0, 255, 0), 2)
            cv.imwrite(str(results_dir / f"{image_stem}_overlay.jpg"), overlay)

        sorted_plain_indices = sort_boxes_reading_order(boxes_plain)

        # Match sorted plain boxes to refined boxes once, based on IoU.
        used_refined_indices = set()
        ordered_refined_indices = []
        for plain_index in sorted_plain_indices:
            plain_box = boxes_plain[plain_index]
            best_refined_idx = None
            best_iou = 0.0
            for refined_idx, refined_box in enumerate(boxes_refined):
                if refined_idx in used_refined_indices:
                    continue
                iou = compute_axis_aligned_iou(plain_box, refined_box)
                if iou > best_iou:
                    best_iou = iou
                    best_refined_idx = refined_idx

            if best_refined_idx is not None:
                used_refined_indices.add(best_refined_idx)
                ordered_refined_indices.append(best_refined_idx)

        if not ordered_refined_indices:
            ordered_refined_indices = list(range(len(boxes_refined)))

        written_paths = []

        # 1) crop line regions, 2) detect words in each line, 3) crop words.
        for line_rank, refined_idx in enumerate(ordered_refined_indices, start=1):
            line_crop = perspective_crop_from_quad(image_bgr, boxes_refined[refined_idx])
            if line_crop is None:
                continue

            cv.imwrite(str(line_crops_dir / f"{image_stem}_line{line_rank}.png"), line_crop)

            word_boxes, _, _, _ = detect_text_regions(craft_detector, line_crop, args, refine_net=None)
            if word_boxes is None or len(word_boxes) == 0:
                fallback_path = crops_dir / f"{image_stem}_line{line_rank}_word1.png"
                cv.imwrite(str(fallback_path), line_crop)
                written_paths.append(fallback_path)
                continue

            sorted_word_indices = sort_boxes_reading_order(word_boxes)
            for word_rank, word_index in enumerate(sorted_word_indices, start=1):
                word_crop = perspective_crop_from_quad(line_crop, word_boxes[word_index])
                if word_crop is None:
                    continue

                output_path = crops_dir / f"{image_stem}_line{line_rank}_word{word_rank}.png"
                cv.imwrite(str(output_path), word_crop)
                written_paths.append(output_path)

        print(f"{image_path} -> saved {len(written_paths)} crops")
        total_crops_written += len(written_paths)

    print(f"Total crops written: {total_crops_written}")

    recognition_results = recognize_crop_folder(recognizer, label_converter, device, crops_dir, args)

    recognized_output_path = results_dir / "recognized.txt"
    with open(recognized_output_path, "w", encoding="utf-8") as result_file:
        for crop_path, recognized_text in recognition_results:
            result_file.write(f"{crop_path}\t{recognized_text}\n")
    print(f"Recognition results saved to {recognized_output_path}")

    write_merged_outputs(recognition_results, results_dir)


if __name__ == "__main__":
    main()
