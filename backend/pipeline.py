import json
import time
from pathlib import Path

import cv2
import numpy as np

from .config import (
    ARTIFACT_DIR, CLASS_COLORS, CLASS_NAMES, CORRECTED_WEIGHT, DECISION_THRESHOLD,
    MINIMUM_CROPS, REVERSAL_WEIGHT,
)
from .model import Classifier
from .segmentation import extract_crops


class AnalysisPipeline:
    def __init__(self, checkpoint_path: Path):
        self.classifier = Classifier(checkpoint_path)

    def analyze(self, image: np.ndarray, job_id: str) -> dict:
        started = time.perf_counter()
        crops, line_boxes, _ = extract_crops(image)
        predictions = self.classifier.predict(crops)
        counts = {name: sum(row["predicted_class"] == name for row in predictions) for name in CLASS_NAMES}
        usable = len(predictions)
        score = None
        status = "insufficient_data"
        dyslexic = None
        if usable >= MINIMUM_CROPS:
            score = (CORRECTED_WEIGHT * counts["Corrected"] + REVERSAL_WEIGHT * counts["Reversal"]) / usable
            score = round(min(score, 1.0), 4)
            dyslexic = score >= DECISION_THRESHOLD
            status = "screening_complete"

        output_dir = ARTIFACT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=False)
        annotated = image.copy()
        for row in predictions:
            color_rgb = CLASS_COLORS[row["predicted_class"]]
            color = (color_rgb[2], color_rgb[1], color_rgb[0])
            cv2.rectangle(annotated, (row["x1"], row["y1"]), (row["x2"], row["y2"]), color, 2)
        annotated_path = output_dir / "annotated.png"
        if not cv2.imwrite(str(annotated_path), annotated):
            raise RuntimeError("Failed to write annotated result.")
        result = {
            "job_id": job_id, "status": status, "screening_result": "elevated_risk" if dyslexic else ("lower_risk" if dyslexic is False else None),
            "dyslexic": dyslexic, "screening_score": score, "decision_threshold": DECISION_THRESHOLD,
            "disclaimer": "This is a research screening result, not a medical diagnosis.",
            "line_count": len(line_boxes), "crop_count": usable, "class_counts": counts,
            "artifacts": {"annotated_image": f"/artifacts/{job_id}/annotated.png"},
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
