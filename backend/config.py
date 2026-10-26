import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
PORT = 2000
ARTIFACT_DIR = Path(os.getenv("DYSLEXIA_ARTIFACT_DIR", ROOT_DIR / "artifacts"))

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
CORRECTED_WEIGHT = 0.7
REVERSAL_WEIGHT = 1.0
DECISION_THRESHOLD = 0.35
MINIMUM_CROPS = 5
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Model selection is intentionally a source-code setting rather than an
# environment variable, so the active model has one unambiguous source.
ACTIVE_MODEL = "vit_b16"

# "auto" uses CUDA when it is available and otherwise uses CPU. Set this to
# "cpu" or "cuda" to require a specific device; an unavailable selection is an
# error rather than an implicit fallback.
MODEL_DEVICE = "auto"

# These are image-level classifiers. The existing segmentation stage supplies
# the handwriting crops that are classified by the selected model.
MODEL_CONFIGS = {
    "model_d_v4": {
        "display_name": "Model D v4",
        "architecture": "model_d_v4",
        "checkpoint_path": MODEL_DIR / "model_d_v4_best.pth",
        "input_size": IMAGE_SIZE,
        "channels": 1,
        "preprocessing": "model_d_v4",
        "class_metadata_key": "class_names",
        "image_size_metadata_key": "img_size",
        "inference_batch_size": 0,
        "trusted_full_deserialization": False,
    },
    "vit_b16": {
        "display_name": "ViT-B/16 Dyslexia Classifier",
        "architecture": "vit_b16",
        "checkpoint_path": MODEL_DIR / "best_vit_b16_dyslexia.pth",
        "input_size": 224,
        "channels": 3,
        "preprocessing": "vit_b16",
        "class_metadata_key": "classes",
        "image_size_metadata_key": None,
        "inference_batch_size": 8,
        # This checkpoint stores a NumPy scalar (best_val_f1) alongside its
        # weights. Torch 2.4 cannot weights-only load that global, so the
        # checked-in, trusted checkpoint is explicitly allowlisted for full
        # deserialization below.
        "trusted_full_deserialization": True,
    },
    "mobilenet_v3_small": {
        "display_name": "MobileNetV3-Small Dyslexia Classifier",
        "architecture": "mobilenet_v3_small",
        "checkpoint_path": MODEL_DIR / "model_a_mobilenet_v3_small.pth",
        "input_size": 128,
        "channels": 1,
        "preprocessing": "mobilenet_v3_small",
        "class_metadata_key": "class_names",
        "image_size_metadata_key": None,
        "inference_batch_size": 64,
        "trusted_full_deserialization": False,
    },
}

SUPPORTED_MODELS = tuple(MODEL_CONFIGS)
SUPPORTED_ARCHITECTURES = ("model_d_v4", "vit_b16", "mobilenet_v3_small")
SUPPORTED_PREPROCESSING = ("model_d_v4", "vit_b16", "mobilenet_v3_small")
_REQUIRED_MODEL_KEYS = {
    "display_name",
    "architecture",
    "checkpoint_path",
    "input_size",
    "channels",
    "preprocessing",
    "class_metadata_key",
    "image_size_metadata_key",
    "inference_batch_size",
    "trusted_full_deserialization",
}


class ConfigurationError(ValueError):
    """Raised when the backend model configuration is invalid."""


def _validate_model_registry() -> None:
    if MODEL_DEVICE not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError(
            f"MODEL_DEVICE must be 'auto', 'cpu', or 'cuda'; got {MODEL_DEVICE!r}."
        )
    for model_id, config in MODEL_CONFIGS.items():
        missing = _REQUIRED_MODEL_KEYS.difference(config)
        if missing:
            raise ConfigurationError(
                f"Model configuration {model_id!r} is missing: "
                f"{', '.join(sorted(missing))}."
            )
        if config["architecture"] not in SUPPORTED_ARCHITECTURES:
            raise ConfigurationError(
                f"Model {model_id!r} uses unsupported architecture "
                f"{config['architecture']!r}. Supported: "
                f"{', '.join(SUPPORTED_ARCHITECTURES)}."
            )
        if config["preprocessing"] not in SUPPORTED_PREPROCESSING:
            raise ConfigurationError(
                f"Model {model_id!r} uses unsupported preprocessing "
                f"{config['preprocessing']!r}. Supported: "
                f"{', '.join(SUPPORTED_PREPROCESSING)}."
            )
        if not isinstance(config["checkpoint_path"], Path):
            raise ConfigurationError(
                f"Checkpoint path for {model_id!r} must be a pathlib.Path."
            )
        if int(config["input_size"]) <= 0 or int(config["channels"]) <= 0:
            raise ConfigurationError(
                f"Model {model_id!r} has an invalid input size or channel count."
            )
        if int(config["inference_batch_size"]) < 0:
            raise ConfigurationError(
                f"Model {model_id!r} inference_batch_size cannot be negative."
            )


def get_model_config(model_id: str | None = None) -> dict:
    """Return a validated model configuration by stable model ID."""
    selected = ACTIVE_MODEL if model_id is None else model_id
    if selected not in MODEL_CONFIGS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ConfigurationError(
            f"Unknown model ID {selected!r} (ACTIVE_MODEL={ACTIVE_MODEL!r}). "
            f"Supported model IDs: {supported}."
        )
    config = dict(MODEL_CONFIGS[selected])
    config["model_id"] = selected
    config["class_names"] = CLASS_NAMES
    return config


def get_active_model_config() -> dict:
    return get_model_config(ACTIVE_MODEL)


_validate_model_registry()

# Fail at import time if ACTIVE_MODEL is unknown or the registry itself is
# malformed. Checkpoint existence is checked when the model is loaded and by
# the /ready endpoint so a missing asset produces an actionable 503.
get_active_model_config()
