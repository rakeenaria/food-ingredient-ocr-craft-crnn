import argparse
import string
import os
import stat
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

DEFAULT_BASIC_CHARACTER_SET = "0123456789abcdefghijklmnopqrstuvwxyz"
SENSITIVE_CHARACTER_SET = string.printable[:-6]


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


def perspective_crop_line_from_quads(
    image_bgr: np.ndarray,
    line_quads,
    pad_along_ratio: float = 0.08,
    pad_cross_ratio: float = 0.04,
):
    """
    Crop a whole text line with orientation-aware perspective transform.
    This avoids axis-aligned line crops that mix neighboring slanted lines.
    """
    if not line_quads:
        return None

    all_points = []
    ordered_quads = []
    for quad in line_quads:
        quad_points = np.array(quad, dtype=np.float32).reshape(-1, 2)
        if quad_points.shape[0] >= 4:
            ordered_quad = order_quad_points(quad_points[:4])
            ordered_quads.append(ordered_quad)
            all_points.append(ordered_quad)
    if not all_points:
        return None

    all_points = np.concatenate(all_points, axis=0).astype(np.float32)
    line_angle = estimate_global_text_angle(line_quads)
    cos_a = float(np.cos(line_angle))
    sin_a = float(np.sin(line_angle))
    along_unit = np.array([cos_a, sin_a], dtype=np.float32)
    cross_unit = np.array([-sin_a, cos_a], dtype=np.float32)

    along_proj = all_points @ along_unit
    if along_proj.size >= 8:
        min_along, max_along = np.percentile(along_proj, [1, 99]).astype(float)
    else:
        min_along, max_along = float(along_proj.min()), float(along_proj.max())

    # Keep line thickness tight around estimated baseline/height to avoid bleeding
    # neighboring lines when detections are slightly tall.
    word_bottoms = []
    word_heights = []
    for quad in ordered_quads:
        cross_vals = quad @ cross_unit
        word_bottoms.append(float(np.max(cross_vals)))
        word_heights.append(max(2.0, float(np.max(cross_vals) - np.min(cross_vals))))

    baseline_cross = float(np.median(word_bottoms)) if word_bottoms else float(np.max(all_points @ cross_unit))
    median_height = float(np.median(word_heights)) if word_heights else 0.0
    raw_width = max_along - min_along
    raw_height = median_height
    if raw_width < 2.0 or raw_height < 2.0:
        return None

    pad_along = max(2.0, raw_width * pad_along_ratio)
    min_cross = baseline_cross - max(2.0, 1.05 * median_height)
    max_cross = baseline_cross + max(1.0, 0.12 * median_height)
    pad_cross = max(1.0, median_height * pad_cross_ratio)
    min_along -= pad_along
    max_along += pad_along
    min_cross -= pad_cross
    max_cross += pad_cross

    line_corners = np.array(
        [
            along_unit * min_along + cross_unit * min_cross,
            along_unit * max_along + cross_unit * min_cross,
            along_unit * max_along + cross_unit * max_cross,
            along_unit * min_along + cross_unit * max_cross,
        ],
        dtype=np.float32,
    )

    target_width = int(round(max_along - min_along))
    target_height = int(round(max_cross - min_cross))
    if target_width < 2 or target_height < 2:
        return None

    destination = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    transform_matrix = cv.getPerspectiveTransform(line_corners, destination)
    cropped_line = cv.warpPerspective(
        image_bgr,
        transform_matrix,
        (target_width, target_height),
        flags=cv.INTER_CUBIC,
        borderMode=cv.BORDER_REPLICATE,
    )
    return cropped_line, transform_matrix.astype(np.float32)


def trim_line_crop_by_ink(line_crop: np.ndarray):
    """
    Trim top/bottom noise by keeping the dominant text band near image center.
    Helps remove slight bleed from adjacent lines in dense layouts.
    """
    if line_crop is None or line_crop.size == 0:
        return line_crop, 0

    gray = cv.cvtColor(line_crop, cv.COLOR_BGR2GRAY)
    _, binary_inv = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    row_ink = binary_inv.sum(axis=1).astype(np.float32)
    if row_ink.size == 0 or float(row_ink.max()) <= 0:
        return line_crop, 0

    active_rows = np.where(row_ink > row_ink.max() * 0.1)[0]
    if active_rows.size == 0:
        return line_crop, 0

    groups = []
    start_row = int(active_rows[0])
    prev_row = int(active_rows[0])
    for row_idx in active_rows[1:]:
        row_idx = int(row_idx)
        if row_idx != prev_row + 1:
            groups.append((start_row, prev_row))
            start_row = row_idx
        prev_row = row_idx
    groups.append((start_row, prev_row))

    crop_center = (line_crop.shape[0] - 1) / 2.0
    best_group = None
    best_score = None
    for y0, y1 in groups:
        group_center = (y0 + y1) / 2.0
        group_height = (y1 - y0 + 1)
        # Prefer larger group and group close to crop center.
        score = group_height - 0.2 * abs(group_center - crop_center)
        if best_score is None or score > best_score:
            best_score = score
            best_group = (y0, y1)

    if best_group is None:
        return line_crop, 0

    y0, y1 = best_group
    margin = max(1, int(round(0.05 * line_crop.shape[0])))
    y0 = max(0, y0 - margin)
    y1 = min(line_crop.shape[0] - 1, y1 + margin)
    trimmed = line_crop[y0 : y1 + 1, :]
    if trimmed.size > 0:
        return trimmed, y0
    return line_crop, 0


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


def select_word_indices_for_line(
    line_quads,
    word_boxes,
    used_indices=None,
    expand_y_ratio: float = 0.18,
    expand_x_ratio: float = 0.02,
    min_iou: float = 0.03,
):
    """
    Select plain word boxes that belong to one line region.
    This keeps per-word recognition even when refiner merges neighboring words.
    """
    if line_quads is None or word_boxes is None:
        return []
    if len(line_quads) == 0 or len(word_boxes) == 0:
        return []

    if used_indices is None:
        used_indices = set()

    line_points = np.concatenate([np.array(quad, dtype=np.float32).reshape(-1, 2) for quad in line_quads], axis=0)
    min_x = float(line_points[:, 0].min())
    max_x = float(line_points[:, 0].max())
    min_y = float(line_points[:, 1].min())
    max_y = float(line_points[:, 1].max())
    line_h = max(1.0, max_y - min_y)
    line_w = max(1.0, max_x - min_x)

    expand_y = expand_y_ratio * line_h
    expand_x = expand_x_ratio * line_w
    line_quad = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]], dtype=np.float32)

    candidates = []
    for idx, word_box in enumerate(word_boxes):
        if idx in used_indices:
            continue
        word_quad = np.array(word_box, dtype=np.float32).reshape(-1, 2)
        if word_quad.shape[0] < 4:
            continue

        word_min_x = float(word_quad[:, 0].min())
        word_max_x = float(word_quad[:, 0].max())
        word_min_y = float(word_quad[:, 1].min())
        word_max_y = float(word_quad[:, 1].max())
        center_x = (word_min_x + word_max_x) * 0.5
        center_y = (word_min_y + word_max_y) * 0.5

        center_in_line = (
            (min_x - expand_x) <= center_x <= (max_x + expand_x)
            and (min_y - expand_y) <= center_y <= (max_y + expand_y)
        )
        iou = compute_axis_aligned_iou(line_quad, word_quad)
        # Keep only words that are actually inside the current line region.
        if center_in_line and iou >= min_iou:
            candidates.append((center_x, idx))

    candidates.sort(key=lambda item: item[0])
    return [idx for _, idx in candidates]


def quad_overlap_ratio_in_image(quad: np.ndarray, width: int, height: int) -> float:
    """Return overlap ratio between quad AABB and image bounds."""
    if width <= 1 or height <= 1:
        return 0.0
    min_x = float(np.min(quad[:, 0]))
    max_x = float(np.max(quad[:, 0]))
    min_y = float(np.min(quad[:, 1]))
    max_y = float(np.max(quad[:, 1]))
    box_w = max(0.0, max_x - min_x)
    box_h = max(0.0, max_y - min_y)
    box_area = box_w * box_h
    if box_area <= 0.0:
        return 0.0

    inter_x0 = max(0.0, min_x)
    inter_y0 = max(0.0, min_y)
    inter_x1 = min(float(width - 1), max_x)
    inter_y1 = min(float(height - 1), max_y)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    return (inter_w * inter_h) / box_area


def is_low_information_crop(crop: np.ndarray) -> bool:
    """Reject almost-empty/black crops that often decode to repeated junk tokens."""
    if crop is None or crop.size == 0:
        return True
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    if float(gray.std()) < 4.0:
        return True
    dark_ratio = float((gray < 15).sum()) / float(gray.size)
    if dark_ratio > 0.97:
        return True
    return False


def estimate_global_text_angle(boxes) -> float:
    """
    Estimate dominant text-line angle in radians from box top/bottom edges.
    Returned angle is normalized to [-pi/2, pi/2].
    """
    if len(boxes) == 0:
        return 0.0

    angles = []
    weights = []
    for box in boxes:
        quad = np.array(box, dtype=np.float32).reshape(-1, 2)
        if quad.shape[0] < 4:
            continue
        ordered = order_quad_points(quad[:4])
        edge_vectors = (ordered[1] - ordered[0], ordered[2] - ordered[3])
        for edge_vec in edge_vectors:
            edge_length = float(np.linalg.norm(edge_vec))
            if edge_length < 2.0:
                continue
            angle = float(np.arctan2(edge_vec[1], edge_vec[0]))
            if angle > np.pi / 2:
                angle -= np.pi
            elif angle < -np.pi / 2:
                angle += np.pi
            angles.append(angle)
            weights.append(edge_length)

    if not angles:
        return 0.0

    # Weighted median is more robust than mean for mixed/noisy box orientations.
    order = np.argsort(angles)
    sorted_angles = np.array(angles)[order]
    sorted_weights = np.array(weights)[order]
    half_weight = sorted_weights.sum() * 0.5
    cumulative = np.cumsum(sorted_weights)
    median_idx = int(np.searchsorted(cumulative, half_weight))
    dominant_angle = float(sorted_angles[min(median_idx, len(sorted_angles) - 1)])

    # Guard against pathological estimates; this line-grouper is for near-horizontal text.
    if abs(dominant_angle) > np.deg2rad(45):
        return 0.0
    return dominant_angle


def group_boxes_by_lines(boxes, min_tol: float = 10.0):
    """
    Group detected boxes into reading lines, each line sorted left-to-right.
    Uses a global deskew angle so slightly rotated text keeps correct line order.
    """
    if len(boxes) == 0:
        return []

    dominant_angle = estimate_global_text_angle(boxes)
    cos_a = float(np.cos(dominant_angle))
    sin_a = float(np.sin(dominant_angle))

    def rotate_point(point_xy):
        x_val = float(point_xy[0])
        y_val = float(point_xy[1])
        # Rotate by -dominant_angle to make text lines closer to horizontal.
        x_rot = x_val * cos_a + y_val * sin_a
        y_rot = -x_val * sin_a + y_val * cos_a
        return x_rot, y_rot

    stats = []  # (index, min_x, baseline_y, height)
    for index, box in enumerate(boxes):
        quad = np.array(box, dtype=np.float32).reshape(-1, 2)
        rotated_points = np.array([rotate_point(point) for point in quad], dtype=np.float32)
        xs = rotated_points[:, 0]
        ys = rotated_points[:, 1]
        min_x = float(xs.min())
        height = float(ys.max() - ys.min())
        ys_sorted = np.sort(ys)
        baseline_y = float(np.mean(ys_sorted[-2:])) if ys_sorted.shape[0] >= 2 else float(ys.max())
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

    for line in grouped_lines:
        line.sort(key=lambda item: item[1])

    return [[item[0] for item in line] for line in grouped_lines]


def sort_boxes_reading_order(boxes, min_tol: float = 10.0):
    """Sort boxes top-to-bottom then left-to-right using baseline-based line grouping."""
    grouped_lines = group_boxes_by_lines(boxes, min_tol=min_tol)
    order = []
    for line in grouped_lines:
        order.extend(line)
    return order


def _sample_link_strength_between_boxes(
    left_box: np.ndarray,
    right_box: np.ndarray,
    score_link: np.ndarray,
    scale_x: float,
    scale_y: float,
):
    """Estimate link strength between two boxes by sampling a thick line in linkmap."""
    left_center = np.mean(left_box, axis=0)
    right_center = np.mean(right_box, axis=0)

    x0 = int(np.clip(round(left_center[0] * scale_x), 0, score_link.shape[1] - 1))
    y0 = int(np.clip(round(left_center[1] * scale_y), 0, score_link.shape[0] - 1))
    x1 = int(np.clip(round(right_center[0] * scale_x), 0, score_link.shape[1] - 1))
    y1 = int(np.clip(round(right_center[1] * scale_y), 0, score_link.shape[0] - 1))

    h_left = max(1.0, float(left_box[:, 1].max() - left_box[:, 1].min()))
    h_right = max(1.0, float(right_box[:, 1].max() - right_box[:, 1].min()))
    thickness = max(1, int(round(0.2 * min(h_left, h_right) * scale_y)))

    mask = np.zeros_like(score_link, dtype=np.uint8)
    cv.line(mask, (x0, y0), (x1, y1), 255, thickness=thickness)
    region = score_link[mask > 0]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def group_boxes_by_refiner_linkmap(boxes, score_link: np.ndarray, image_shape, min_tol: float = 10.0):
    """
    Group word boxes into lines using refiner-enhanced linkmap connectivity.
    Boxes stay word-level; linkmap is only used for ordering/grouping.
    """
    if boxes is None or len(boxes) == 0:
        return []
    if score_link is None or score_link.size == 0:
        return group_boxes_by_lines(boxes, min_tol=min_tol)

    image_h, image_w = image_shape[:2]
    if image_h <= 0 or image_w <= 0:
        return group_boxes_by_lines(boxes, min_tol=min_tol)

    scale_x = float(score_link.shape[1]) / float(image_w)
    scale_y = float(score_link.shape[0]) / float(image_h)

    dominant_angle = estimate_global_text_angle(boxes)
    cos_a = float(np.cos(dominant_angle))
    sin_a = float(np.sin(dominant_angle))

    def rotate_xy(x_val: float, y_val: float):
        x_rot = x_val * cos_a + y_val * sin_a
        y_rot = -x_val * sin_a + y_val * cos_a
        return x_rot, y_rot

    centers = []
    heights = []
    widths = []
    for box in boxes:
        quad = np.array(box, dtype=np.float32).reshape(-1, 2)
        center = quad.mean(axis=0)
        cx, cy = float(center[0]), float(center[1])
        rx, ry = rotate_xy(cx, cy)
        centers.append((cx, cy, rx, ry))
        heights.append(max(1.0, float(quad[:, 1].max() - quad[:, 1].min())))
        widths.append(max(1.0, float(quad[:, 0].max() - quad[:, 0].min())))

    median_h = float(np.median(heights)) if heights else min_tol
    median_w = float(np.median(widths)) if widths else 20.0
    line_tol = max(min_tol, 0.7 * median_h)

    parent = list(range(len(boxes)))

    def find(x_idx):
        while parent[x_idx] != x_idx:
            parent[x_idx] = parent[parent[x_idx]]
            x_idx = parent[x_idx]
        return x_idx

    def union(a_idx, b_idx):
        ra = find(a_idx)
        rb = find(b_idx)
        if ra != rb:
            parent[rb] = ra

    link_thr = 0.20
    for i in range(len(boxes)):
        _, _, rx_i, ry_i = centers[i]
        quad_i = np.array(boxes[i], dtype=np.float32).reshape(-1, 2)
        for j in range(i + 1, len(boxes)):
            _, _, rx_j, ry_j = centers[j]
            if abs(ry_i - ry_j) > line_tol:
                continue
            if abs(rx_i - rx_j) > max(8.0 * median_w, 120.0):
                continue

            quad_j = np.array(boxes[j], dtype=np.float32).reshape(-1, 2)
            if rx_i <= rx_j:
                left_box, right_box = quad_i, quad_j
            else:
                left_box, right_box = quad_j, quad_i

            link_score = _sample_link_strength_between_boxes(
                left_box,
                right_box,
                score_link=score_link,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            if link_score >= link_thr:
                union(i, j)

    groups = defaultdict(list)
    for idx in range(len(boxes)):
        groups[find(idx)].append(idx)

    global_slope = float(np.tan(dominant_angle))
    lines = []
    for indices in groups.values():
        indices.sort(key=lambda idx: centers[idx][2])  # sort by rotated x
        rx_values = [centers[idx][2] for idx in indices]
        # Base-line key that compensates global tilt so slanted lines keep order.
        base_values = [centers[idx][1] - global_slope * centers[idx][0] for idx in indices]
        line_base = float(np.median(base_values))
        lines.append(
            {
                "base": line_base,
                "min_rx": float(min(rx_values)),
                "max_rx": float(max(rx_values)),
                "indices": indices,
            }
        )

    lines.sort(key=lambda item: (item["base"], item["min_rx"]))

    # Merge fragmented components that likely belong to the same slanted text line.
    merged_lines = []
    base_tol = max(min_tol, 0.45 * median_h)
    gap_tol = max(20.0, 1.8 * median_w)
    for line in lines:
        if not merged_lines:
            merged_lines.append(line)
            continue
        prev = merged_lines[-1]
        base_close = abs(line["base"] - prev["base"]) <= base_tol
        x_gap = line["min_rx"] - prev["max_rx"]
        close_in_x = x_gap <= gap_tol
        if base_close and close_in_x:
            combined = sorted(prev["indices"] + line["indices"], key=lambda idx: centers[idx][2])
            prev["indices"] = combined
            prev["max_rx"] = max(prev["max_rx"], line["max_rx"])
            prev["base"] = float(
                np.median([centers[idx][1] - global_slope * centers[idx][0] for idx in combined])
            )
        else:
            merged_lines.append(line)

    merged_lines.sort(key=lambda item: (item["base"], item["min_rx"]))
    return [line["indices"] for line in merged_lines]


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


def expected_num_classes(prediction_name: str, character_count: int) -> int:
    token_count = 2 if "Attn" in prediction_name else 1
    return character_count + token_count


def checkpoint_num_classes(saved_model_path: str):
    """Return output class count encoded in checkpoint head weights, if available."""
    state = torch.load(saved_model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        return None

    keys = (
        "Prediction.generator.weight",
        "module.Prediction.generator.weight",
        "Prediction.weight",
        "module.Prediction.weight",
    )
    for key in keys:
        if key in state:
            return int(state[key].shape[0])
    return None


def normalize_character_configuration(args):
    """
    Resolve recognizer character set from CLI flags and checkpoint head shape.
    Automatically enables sensitive mode for known 96-class checkpoints.
    """
    if args.sensitive:
        args.character = SENSITIVE_CHARACTER_SET
        return

    current_num_classes = expected_num_classes(args.Prediction, len(args.character))
    checkpoint_classes = checkpoint_num_classes(args.saved_model)
    if checkpoint_classes is None or checkpoint_classes == current_num_classes:
        return

    sensitive_num_classes = expected_num_classes(args.Prediction, len(SENSITIVE_CHARACTER_SET))
    if args.character == DEFAULT_BASIC_CHARACTER_SET and checkpoint_classes == sensitive_num_classes:
        args.sensitive = True
        args.character = SENSITIVE_CHARACTER_SET
        print(
            f"Info: checkpoint expects {checkpoint_classes} classes. "
            "Enabling sensitive character set automatically."
        )
        return

    raise ValueError(
        f"Checkpoint/output class mismatch: checkpoint has {checkpoint_classes} classes, "
        f"but current configuration expects {current_num_classes}. "
        "Pass --sensitive or set --character to match the training charset."
    )


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


def write_merged_outputs(recognition_results, results_dir: Path, merged_output_path: Path):
    """Reconstruct paragraph-level text in reading order and write output files."""
    grouped_predictions = defaultdict(list)

    for image_path, predicted_text in recognition_results:
        stem_name = Path(image_path).stem
        # Skip line-only crops to avoid duplicate text in final merge.
        if "_line" in stem_name and "_word" not in stem_name:
            continue

        base_name, order_index = parse_word_order_index(stem_name)
        grouped_predictions[base_name].append((order_index, predicted_text))

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
    parser.add_argument("--merged_output", default="./outputs/merged.txt", help="path to merged paragraph output")
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


def prepare_output_directory(output_dir: Path, label: str) -> Path:
    """
    Ensure output directory is empty.
    If cleanup fails because the directory is locked, stop with a clear error.
    """
    if not output_dir.exists():
        return output_dir

    def remove_readonly(action, path, exc_info):
        _ = exc_info
        try:
            os.chmod(path, stat.S_IWRITE)
        except Exception:
            pass
        action(path)

    try:
        os.chmod(output_dir, stat.S_IWRITE)
        shutil.rmtree(output_dir, onerror=remove_readonly)
        return output_dir
    except PermissionError as error:
        raise RuntimeError(
            f"Failed to remove {label} folder ({output_dir}). "
            "Close files/apps that lock this folder, then run again."
        ) from error


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    normalize_character_configuration(args)

    input_dir = Path(args.input_folder)
    crops_dir = Path(args.crops_folder)
    results_dir = Path(args.results_folder)
    merged_output_path = Path(args.merged_output)

    # Rebuild outputs from scratch for each run to avoid stale predictions.
    crops_dir = prepare_output_directory(crops_dir, "crops")
    results_dir = prepare_output_directory(results_dir, "results")
    crops_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)

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

        # Refiner output is used for line-level layout, plain CRAFT output for
        # word-level crops so recognizer still sees per-word images.
        if args.refine and refiner is not None:
            boxes_for_layout = boxes_refined
            boxes_for_words, _, _, _ = detect_text_regions(craft_detector, image_bgr, args, refine_net=None)
            if boxes_for_words is None or len(boxes_for_words) == 0:
                boxes_for_words = boxes_for_layout
        else:
            boxes_for_layout, _, _, _ = detect_text_regions(craft_detector, image_bgr, args, refine_net=None)
            if boxes_for_layout is None or len(boxes_for_layout) == 0:
                boxes_for_layout = boxes_refined
            boxes_for_words = boxes_for_layout

        if boxes_for_layout is None or len(boxes_for_layout) == 0:
            print(f"{image_path} -> 0 boxes")
            continue

        if args.save_overlay:
            overlay = image_bgr.copy()
            polygons_for_draw = (
                polygons_refined
                if polygons_refined is not None and len(polygons_refined) == len(boxes_refined)
                else boxes_for_layout
            )
            for polygon in polygons_for_draw:
                points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv.polylines(overlay, [points], True, (0, 255, 0), 2)
            cv.imwrite(str(results_dir / f"{image_stem}_overlay.jpg"), overlay)

        if args.refine and refiner is not None:
            grouped_line_indices = group_boxes_by_refiner_linkmap(
                boxes_for_words,
                score_link=score_link,
                image_shape=image_bgr.shape,
            )
        else:
            grouped_line_indices = group_boxes_by_lines(boxes_for_words)
        if len(grouped_line_indices) == 0:
            print(f"{image_path} -> 0 boxes")
            continue

        written_paths = []
        for line_rank, line_indices in enumerate(grouped_line_indices, start=1):
            for word_rank, word_index in enumerate(line_indices, start=1):
                word_quad_global = np.array(boxes_for_words[word_index], dtype=np.float32).reshape(-1, 2)
                word_crop = perspective_crop_from_quad(image_bgr, word_quad_global)
                if word_crop is None or word_crop.size == 0:
                    word_min_x = max(0, int(np.floor(word_quad_global[:, 0].min())))
                    word_min_y = max(0, int(np.floor(word_quad_global[:, 1].min())))
                    word_max_x = min(image_bgr.shape[1] - 1, int(np.ceil(word_quad_global[:, 0].max())))
                    word_max_y = min(image_bgr.shape[0] - 1, int(np.ceil(word_quad_global[:, 1].max())))
                    if word_max_x <= word_min_x or word_max_y <= word_min_y:
                        continue
                    word_crop = image_bgr[word_min_y : word_max_y + 1, word_min_x : word_max_x + 1]
                    if word_crop.size == 0:
                        continue
                if is_low_information_crop(word_crop):
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

    legacy_merged_path = results_dir / "merged.txt"
    if legacy_merged_path.resolve() != merged_output_path.resolve() and legacy_merged_path.exists():
        legacy_merged_path.unlink()

    write_merged_outputs(recognition_results, results_dir, merged_output_path)


if __name__ == "__main__":
    main()
