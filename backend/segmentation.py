from dataclasses import dataclass
import logging

import cv2
import numpy as np
from PIL import Image

from .config import MIN_AREA, MIN_HEIGHT, PADDING

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Crop:
    image: np.ndarray
    x1: int
    y1: int
    x2: int
    y2: int
    crop_id: int


def remove_horizontal_rules(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = ink.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(35, width // 18), 1)
    )
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    rules = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    return cv2.subtract(ink, rules), rules


def segment_lines(gray: np.ndarray, pil_image: Image.Image | None = None) -> list[tuple[int, int, int, int]]:
    height, width = gray.shape[:2]
    try:
        from kraken import binarization, pageseg

        pil_image = pil_image or Image.fromarray(gray)
        segmented = pageseg.segment(binarization.nlbin(pil_image))
        lines = getattr(segmented, "lines", None)
        if lines is None and isinstance(segmented, dict):
            lines = segmented.get("boxes", [])
        if lines is None:
            raise TypeError("Kraken returned an unsupported segmentation result.")
        boxes = []
        for line in lines:
            x1, y1, x2, y2 = map(int, getattr(line, "bbox", line))
            box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        if boxes:
            return sorted(boxes, key=lambda box: box[1])
    except ImportError:
        pass
    except Exception as exc:
        logger.warning(
            "Kraken line segmentation failed; falling back to OpenCV: %s",
            exc,
        )

    threshold = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(40, width // 15), 3)
    )
    line_mask = cv2.dilate(threshold, kernel, iterations=2)
    contours, _ = cv2.findContours(
        line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width > 0.12 * width and box_height > 10:
            boxes.append(
                (
                    max(0, x - 5),
                    max(0, y - 5),
                    min(width, x + box_width + 5),
                    min(height, y + box_height + 5),
                )
            )
    return sorted(boxes, key=lambda box: box[1])


def _split_wide_component(component: np.ndarray) -> list[np.ndarray]:
    if cv2.countNonZero(component) < 30:
        return [component]
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    peak = float(distance.max())
    if peak < 2.5:
        return [component]
    _, foreground = cv2.threshold(
        distance, max(2.0, 0.42 * peak), 255, cv2.THRESH_BINARY
    )
    count, markers = cv2.connectedComponents(np.uint8(foreground))
    if count < 3:
        return [component]
    markers = markers.astype(np.int32)
    markers[component == 0] = 1
    markers[markers > 1] += 1
    cv2.watershed(cv2.cvtColor(component, cv2.COLOR_GRAY2BGR), markers)
    pieces = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        piece = np.uint8(markers == label) * 255
        if cv2.countNonZero(piece) >= 12:
            pieces.append(piece)
    return pieces if len(pieces) >= 2 else [component]


def generate_candidate_boxes(
    cleaned_mask: np.ndarray, page_width: int, typical_line_height: float
) -> list[tuple[int, int, int, int]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, 8)
    boxes = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if area < MIN_AREA or height < MIN_HEIGHT:
            continue
        if width > max(2.5 * height, 0.18 * page_width) and area > 2 * MIN_AREA:
            local = np.uint8(labels[y : y + height, x : x + width] == index) * 255
            pieces = _split_wide_component(local)
            if len(pieces) > 1:
                for piece in pieces:
                    px, py, pw, ph = cv2.boundingRect(piece)
                    if pw > 0 and ph >= MIN_HEIGHT and cv2.countNonZero(piece) >= MIN_AREA:
                        boxes.append((x + px, y + py, x + px + pw, y + py + ph))
                continue
        boxes.append((x, y, x + width, y + height))
    filtered = [
        box
        for box in boxes
        if not (box[2] - box[0] > 0.45 * page_width and
                box[3] - box[1] < 0.20 * typical_line_height)
    ]
    cluster_size = max(typical_line_height * 0.7, 15)
    return sorted(filtered, key=lambda box: (round(box[1] / cluster_size), box[0]))


def extract_crops(image: np.ndarray) -> tuple[list[Crop], list[tuple[int, int, int, int]], np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray)
    line_boxes = segment_lines(gray, pil_image)
    threshold = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    cleaned, _ = remove_horizontal_rules(threshold)
    line_height = np.median([box[3] - box[1] for box in line_boxes]) if line_boxes else max(15, gray.shape[0] / 20)
    boxes = generate_candidate_boxes(cleaned, gray.shape[1], float(line_height))
    crops = []
    for crop_id, (x1, y1, x2, y2) in enumerate(boxes):
        left, top = max(0, x1 - PADDING), max(0, y1 - PADDING)
        right, bottom = min(image.shape[1], x2 + PADDING), min(image.shape[0], y2 + PADDING)
        crops.append(Crop(image[top:bottom, left:right].copy(), left, top, right, bottom, crop_id))
    return crops, line_boxes, cleaned
