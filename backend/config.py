from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.getenv("DYSLEXIA_ARTIFACT_DIR", ROOT_DIR / "artifacts"))
CHECKPOINT_PATH = Path(
    os.getenv(
        "DYSLEXIA_CHECKPOINT",
        ROOT_DIR / "model_d_v4_best.pth.zip",
    )
)

CLASS_NAMES = ("Corrected", "Normal", "Reversal")
CLASS_COLORS = {
    "Normal": (46, 204, 113),
    "Corrected": (243, 156, 18),
    "Reversal": (231, 76, 60),
}
IMAGE_SIZE = 160
MIN_AREA = 18
MIN_HEIGHT = 7
PADDING = 3
CORRECTED_WEIGHT = 0.5
REVERSAL_WEIGHT = 1.0
DECISION_THRESHOLD = 0.6
MINIMUM_CROPS = 5
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
