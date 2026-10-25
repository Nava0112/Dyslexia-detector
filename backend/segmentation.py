import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import MIN_AREA, MIN_HEIGHT, PADDING


@dataclass(frozen=True)
class Crop:
    image: np.ndarray
    x1: int
    y1: int
    x2: int
    y2: int
    crop_id: int
    line_id: int = 0


def validate_image(image: np.ndarray) -> None:
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image: input must be a non-null numpy ndarray.")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError(f"Invalid image shape: {getattr(image, 'shape', None)}.")
    if image.shape[0] < 15 or image.shape[1] < 15:
        raise ValueError(f"Image dimensions too small for analysis: {image.shape[:2]}.")


def remove_horizontal_rules(ink: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects and subtracts notebook ruled lines without erasing character strokes.
    """
    height, width = ink.shape[:2]
    kernel_w = max(35, width // 18)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    rules = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    cleaned = cv2.subtract(ink, rules)
    return cleaned, rules


def binarize_image(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts grayscale image to binary ink mask (ink=255, background=0).
    Checks polarity and removes notebook horizontal rules.
    """
    if np.mean(gray) > 127:
        raw_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    else:
        raw_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cleaned_ink, _ = remove_horizontal_rules(raw_ink)
    return cleaned_ink, raw_ink


def segment_lines(gray: np.ndarray, pil_image: Optional[Image.Image] = None) -> List[Tuple[int, int, int, int]]:
    """
    Detects text-line bounding boxes in reading order (top-to-bottom).
    Uses Kraken if installed, or enhanced OpenCV horizontal projection analysis.
    """
    height, width = gray.shape[:2]
    try:
        from kraken import binarization, pageseg

        pil_img = pil_image or Image.fromarray(gray)
        segmented = pageseg.segment(binarization.nlbin(pil_img))
        boxes = []
        for line in segmented.lines:
            x1, y1, x2, y2 = map(int, line.bbox)
            box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        if boxes:
            return sorted(boxes, key=lambda b: b[1])
    except (ImportError, Exception):
        pass

    # Enhanced OpenCV line segmentation
    cleaned_ink, _ = binarize_image(gray)

    # 1D purely horizontal dilation bridges words along a line without bridging lines vertically
    h_kernel_width = max(25, width // 22)
    kernel_h1 = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 1))
    line_mask = cv2.dilate(cleaned_ink, kernel_h1, iterations=2)

    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw > 0.04 * width and bh >= 6:
            raw_boxes.append([x, y, x + bw, y + bh])

    if not raw_boxes:
        ink_pts = cv2.findNonZero(cleaned_ink)
        if ink_pts is not None:
            x, y, bw, bh = cv2.boundingRect(ink_pts)
            return [(x, y, x + bw, y + bh)]
        return [(0, 0, width, height)]

    raw_boxes.sort(key=lambda b: b[1])

    # Merge horizontally disjoint boxes belonging to the same text line
    merged_lines = []
    for b in raw_boxes:
        if not merged_lines:
            merged_lines.append(b)
            continue
        prev = merged_lines[-1]
        v_overlap = max(0, min(prev[3], b[3]) - max(prev[1], b[1]))
        min_h = min(prev[3] - prev[1], b[3] - b[1])
        if min_h > 0 and (v_overlap / min_h > 0.40):
            merged_lines[-1] = [
                min(prev[0], b[0]),
                min(prev[1], b[1]),
                max(prev[2], b[2]),
                max(prev[3], b[3]),
            ]
        else:
            merged_lines.append(b)

    # Split lines that accidentally merged 2 adjacent lines (height > 1.65 * median)
    line_heights = [b[3] - b[1] for b in merged_lines]
    median_h = float(np.median(line_heights)) if line_heights else 30.0

    refined_lines = []
    for b in merged_lines:
        bh = b[3] - b[1]
        if bh > 1.65 * median_h and bh > 25:
            sub_mask = cleaned_ink[b[1]:b[3], b[0]:b[2]]
            if sub_mask.size > 0:
                row_sums = np.sum(sub_mask == 255, axis=1)
                k = max(3, int(median_h * 0.15))
                smoothed = np.convolve(row_sums, np.ones(k) / k, mode="same")
                mid_start = int(0.25 * bh)
                mid_end = int(0.75 * bh)
                if mid_end > mid_start:
                    min_idx = mid_start + int(np.argmin(smoothed[mid_start:mid_end]))
                    if smoothed[min_idx] < 0.35 * np.max(smoothed):
                        split_y = b[1] + min_idx
                        refined_lines.append([b[0], b[1], b[2], split_y])
                        refined_lines.append([b[0], split_y, b[2], b[3]])
                        continue
        refined_lines.append(b)

    # Add margin and clamp to image bounds
    final_boxes = []
    for b in refined_lines:
        pad_y = max(2, int((b[3] - b[1]) * 0.06))
        pad_x = max(2, int((b[2] - b[0]) * 0.02))
        final_boxes.append((
            max(0, b[0] - pad_x),
            max(0, b[1] - pad_y),
            min(width, b[2] + pad_x),
            min(height, b[3] + pad_y),
        ))

    return sorted(final_boxes, key=lambda b: b[1])


def _split_wide_component(component: np.ndarray, min_height: int = 7) -> List[np.ndarray]:
    """
    Separates touching characters via vertical projection valley detection and watershed.
    Preserves single wide letters like 'm' and 'w'.
    """
    h, w = component.shape
    if w < 16 or h < 10 or cv2.countNonZero(component) < 35:
        return [component]

    proj = np.sum(component == 255, axis=0)
    peak = float(np.max(proj))
    if peak == 0:
        return [component]

    mid_start = max(4, int(w * 0.22))
    mid_end = min(w - 4, int(w * 0.78))
    if mid_end <= mid_start:
        return [component]

    valley_idx = mid_start + int(np.argmin(proj[mid_start:mid_end]))
    valley_val = proj[valley_idx]

    # Only split if valley is significantly lower than peak
    if valley_val <= 0.45 * peak:
        dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        left_sub = dist[:, :valley_idx]
        right_sub = dist[:, valley_idx:]
        if left_sub.size == 0 or right_sub.size == 0:
            return [component]

        left_peak = np.argmax(left_sub)
        right_peak = np.argmax(right_sub)
        left_y, left_x = np.unravel_index(left_peak, left_sub.shape)
        right_y, right_x = np.unravel_index(right_peak, right_sub.shape)
        right_x += valley_idx

        markers = np.zeros((h, w), dtype=np.int32)
        markers[component == 0] = 1
        markers[left_y, left_x] = 2
        markers[right_y, right_x] = 3

        cv2.watershed(cv2.cvtColor(component, cv2.COLOR_GRAY2BGR), markers)
        piece1 = np.uint8(markers == 2) * 255
        piece2 = np.uint8(markers == 3) * 255

        _, _, _, ph1 = cv2.boundingRect(piece1)
        _, _, _, ph2 = cv2.boundingRect(piece2)

        if cv2.countNonZero(piece1) >= 12 and cv2.countNonZero(piece2) >= 12 and ph1 >= min_height and ph2 >= min_height:
            return [piece1, piece2]

    return [component]


def _compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def _resolve_overlaps(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Suppresses duplicate and heavily overlapping character boxes within a line.
    """
    if len(boxes) <= 1:
        return boxes

    sorted_boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    kept: List[Tuple[int, int, int, int]] = []

    for b in sorted_boxes:
        discard = False
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        for idx, k in enumerate(kept):
            iou = _compute_iou(b, k)
            if iou > 0.65:
                area_k = (k[2] - k[0]) * (k[3] - k[1])
                if area_b > area_k:
                    kept[idx] = (min(b[0], k[0]), min(b[1], k[1]), max(b[2], k[2]), max(b[3], k[3]))
                discard = True
                break

            inter_x = max(0, min(b[2], k[2]) - max(b[0], k[0]))
            inter_y = max(0, min(b[3], k[3]) - max(b[1], k[1]))
            inter_area = inter_x * inter_y
            if area_b > 0 and inter_area / area_b > 0.85:
                discard = True
                break
            area_k = (k[2] - k[0]) * (k[3] - k[1])
            if area_k > 0 and inter_area / area_k > 0.85:
                kept[idx] = b
                discard = True
                break

        if not discard:
            kept.append(b)

    return sorted(kept, key=lambda b: b[0])


def extract_crops(
    image: np.ndarray,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> Tuple[List[Crop], List[Tuple[int, int, int, int]], Dict[str, Any]]:
    """
    Extracts individual character crops constrained within text lines in natural reading order.
    Returns:
        crops: List of Crop objects with bounding coordinates and BGR images
        line_boxes: List of line bounding boxes (x1, y1, x2, y2)
        debug_info: Dictionary containing segmentation statistics and diagnostic metadata
    """
    validate_image(image)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    pil_image = Image.fromarray(gray)

    # 1. Line Segmentation
    line_boxes = segment_lines(gray, pil_image)

    # 2. Binarization & Rule Removal
    cleaned_ink, raw_ink = binarize_image(gray)

    # Scale estimation
    line_heights = [b[3] - b[1] for b in line_boxes if b[3] - b[1] >= 8]
    global_median_lh = float(np.median(line_heights)) if line_heights else max(20.0, height / 20.0)

    debug_stats: Dict[str, Any] = {
        "lines_detected": len(line_boxes),
        "median_line_height": round(global_median_lh, 2),
        "total_initial_components": 0,
        "grouped_fragments": 0,
        "split_components": 0,
        "rejected_components": 0,
        "final_crops_count": 0,
        "warnings": [],
        "cleaned_mask": cleaned_ink,
    }

    crops: List[Crop] = []
    crop_counter = 0

    # Process each line independently to prevent vertical cross-line merges
    for line_id, (lx1, ly1, lx2, ly2) in enumerate(line_boxes):
        line_h = max(10, ly2 - ly1)
        line_crop = cleaned_ink[ly1:ly2, lx1:lx2]
        if line_crop.size == 0 or cv2.countNonZero(line_crop) < 10:
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_crop, 8)
        debug_stats["total_initial_components"] += (num_labels - 1)

        # Scale-aware relative thresholds
        min_h = max(5, int(line_h * 0.14))
        min_area = max(8, int((line_h * 0.14) ** 2 * 0.4))
        typical_char_w = line_h * 0.65

        main_components = []
        fragments = []

        for i in range(1, num_labels):
            cx, cy, cw, ch, area = stats[i]

            # Reject borders or long horizontal rules
            if cw > 0.60 * width and ch < 0.25 * line_h:
                debug_stats["rejected_components"] += 1
                continue

            # Split unusually wide components (touching characters)
            if cw > max(1.25 * line_h, 1.8 * typical_char_w) and area > 2 * min_area:
                local_mask = np.uint8(labels[cy:cy + ch, cx:cx + cw] == i) * 255
                pieces = _split_wide_component(local_mask, min_height=min_h)
                if len(pieces) > 1:
                    debug_stats["split_components"] += (len(pieces) - 1)
                    for piece in pieces:
                        px, py, pw, ph = cv2.boundingRect(piece)
                        if pw > 0 and ph >= min_h and cv2.countNonZero(piece) >= min_area:
                            main_components.append([cx + px, cy + py, cx + px + pw, cy + py + ph, cv2.countNonZero(piece)])
                    continue

            if area >= min_area and ch >= min_h:
                main_components.append([cx, cy, cx + cw, cy + ch, area])
            elif area >= 4 and ch >= 3:
                fragments.append([cx, cy, cx + cw, cy + ch, area])
            else:
                debug_stats["rejected_components"] += 1

        # Conservative fragment grouping (e.g. i-dots, accents, or broken strokes)
        grouped_line_boxes = []
        used_fragments = set()

        for comp in main_components:
            cx_center = (comp[0] + comp[2]) / 2.0
            cw = comp[2] - comp[0]
            box = list(comp[:4])

            for f_idx, frag in enumerate(fragments):
                if f_idx in used_fragments:
                    continue
                fx_center = (frag[0] + frag[2]) / 2.0
                if abs(fx_center - cx_center) <= max(5, 0.45 * cw):
                    if 0 <= (comp[1] - frag[3]) <= 0.60 * line_h:
                        box[0] = min(box[0], frag[0])
                        box[1] = min(box[1], frag[1])
                        box[2] = max(box[2], frag[2])
                        box[3] = max(box[3], frag[3])
                        used_fragments.add(f_idx)
                        debug_stats["grouped_fragments"] += 1

            grouped_line_boxes.append((lx1 + box[0], ly1 + box[1], lx1 + box[2], ly1 + box[3]))

        # Suppress duplicate and heavily overlapping boxes
        resolved_boxes = _resolve_overlaps(grouped_line_boxes)

        # Proportional padding based on line height
        pad_size = max(2, int(line_h * 0.05))

        for (gx1, gy1, gx2, gy2) in resolved_boxes:
            x1 = max(0, gx1 - pad_size)
            y1 = max(0, gy1 - pad_size)
            x2 = min(width, gx2 + pad_size)
            y2 = min(height, gy2 + pad_size)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w <= 0 or crop_h <= 0:
                continue

            crop_img = image[y1:y2, x1:x2].copy()
            crop_ink = cleaned_ink[y1:y2, x1:x2]
            if crop_ink.size == 0 or cv2.countNonZero(crop_ink) < 6:
                continue

            crops.append(Crop(
                image=crop_img,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                crop_id=crop_counter,
                line_id=line_id,
            ))
            crop_counter += 1

    debug_stats["final_crops_count"] = len(crops)
    if len(crops) < 5:
        debug_stats["warnings"].append("low_candidate_count_insufficient_signal")

    # Optional Debug Artifacts Generation
    if debug or debug_dir is not None:
        target_dir = Path(debug_dir) if debug_dir else Path("artifacts/segmentation_debug")
        target_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(target_dir / "01_original.png"), image)
        cv2.imwrite(str(target_dir / "02_grayscale.png"), gray)
        cv2.imwrite(str(target_dir / "03_binary_raw.png"), raw_ink)
        cv2.imwrite(str(target_dir / "04_cleaned_ink.png"), cleaned_ink)

        line_overlay = image.copy()
        for idx, (bx1, by1, bx2, by2) in enumerate(line_boxes):
            cv2.rectangle(line_overlay, (bx1, by1), (bx2, by2), (255, 120, 0), 2)
            cv2.putText(line_overlay, f"Line {idx}", (bx1 + 5, by1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
        cv2.imwrite(str(target_dir / "05_line_boxes.png"), line_overlay)

        char_overlay = image.copy()
        for c in crops:
            cv2.rectangle(char_overlay, (c.x1, c.y1), (c.x2, c.y2), (0, 200, 50), 1)
        cv2.imwrite(str(target_dir / "06_character_boxes.png"), char_overlay)

        meta_to_save = {k: v for k, v in debug_stats.items() if k != "cleaned_mask"}
        with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, indent=2)

    return crops, line_boxes, debug_stats
