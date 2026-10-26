"""Active-model loading, preprocessing, and crop classification."""

import logging
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    CLASS_NAMES,
    MODEL_DEVICE,
    ROOT_DIR,
    ConfigurationError,
    get_active_model_config,
    get_model_config,
)
from .model_architectures import ModelDV4, build_architecture

logger = logging.getLogger(__name__)

# Kept as public names for callers that imported the original implementation.
__all__ = [
    "Classifier",
    "ModelDV4",
    "ModelInferenceError",
    "ModelLoadError",
    "preprocess_crop",
]


class ModelLoadError(RuntimeError):
    """Raised when the configured model cannot be constructed or loaded."""


class ModelInferenceError(RuntimeError):
    """Raised when inference fails or returns an invalid result."""


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def require_checkpoint(config: dict) -> Path:
    """Return the configured checkpoint path or raise an actionable error."""
    path = Path(config["checkpoint_path"])
    if not path.is_file():
        raise ModelLoadError(
            f"Checkpoint for model {config['model_id']!r} not found: "
            f"{_relative_path(path)}"
        )
    return path


def resolve_device(preference: str = MODEL_DEVICE) -> torch.device:
    """Resolve the configured device without silently changing preference."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ModelLoadError(
                "MODEL_DEVICE is 'cuda', but CUDA is not available. "
                "Set MODEL_DEVICE to 'auto' or 'cpu' in backend/config.py."
            )
        return torch.device("cuda")
    raise ModelLoadError(
        f"Unsupported MODEL_DEVICE {preference!r}; use 'auto', 'cpu', or 'cuda'."
    )


def _as_gray(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ModelInferenceError("Cannot preprocess an empty crop.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ModelInferenceError(
            f"Unsupported crop shape {image.shape}; expected HxW or HxWx3."
        )
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _square_resize(gray: np.ndarray, target_size: int) -> np.ndarray:
    side = max(gray.shape)
    square = np.zeros((side, side), dtype=gray.dtype)
    y_offset = (side - gray.shape[0]) // 2
    x_offset = (side - gray.shape[1]) // 2
    square[
        y_offset : y_offset + gray.shape[0],
        x_offset : x_offset + gray.shape[1],
    ] = gray
    return cv2.resize(
        square, (target_size, target_size), interpolation=cv2.INTER_AREA
    )


def preprocess_model_d_v4(image: np.ndarray) -> torch.Tensor:
    """Exact Model D v4 preprocessing: 160x160, CLAHE, range [-1, 1]."""
    square = _square_resize(_as_gray(image), 160)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(square)
    normalized = (enhanced.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)


def preprocess_mobilenet_v3_small(image: np.ndarray) -> torch.Tensor:
    """Exact Model A preprocessing: adaptive-thresholded grayscale, 128x128."""
    square = _square_resize(_as_gray(image), 128)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(square)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5,
    )
    normalized = binary.astype(np.float32) / 255.0
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)


def preprocess_vit_b16(image: np.ndarray) -> torch.Tensor:
    """Exact ViT inference preprocessing: 224x224 RGB, ImageNet normalization."""
    square = _square_resize(_as_gray(image), 224)
    rgb = cv2.cvtColor(square, cv2.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


_PREPROCESSORS = {
    "model_d_v4": preprocess_model_d_v4,
    "mobilenet_v3_small": preprocess_mobilenet_v3_small,
    "vit_b16": preprocess_vit_b16,
}


def preprocess_crop(image: np.ndarray, model_id: str | None = None) -> torch.Tensor:
    """Preprocess one crop using the selected model's verified contract."""
    try:
        config = (
            get_active_model_config()
            if model_id is None
            else get_model_config(model_id)
        )
    except ConfigurationError as exc:
        raise ModelLoadError(str(exc)) from exc
    preprocessing = config["preprocessing"]
    try:
        preprocessor = _PREPROCESSORS[preprocessing]
    except KeyError as exc:
        supported = ", ".join(sorted(_PREPROCESSORS))
        raise ModelLoadError(
            f"No preprocessing adapter for {preprocessing!r}. Supported: {supported}."
        ) from exc
    return preprocessor(image)


def _load_checkpoint(path: Path, config: dict) -> dict:
    """Load a trusted checkpoint, preferring PyTorch's weights-only mode."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as safe_error:
        if not config["trusted_full_deserialization"]:
            raise ModelLoadError(
                f"Checkpoint for {config['model_id']!r} cannot be loaded with "
                "weights_only=True and is not allowlisted for full deserialization."
            ) from safe_error
        logger.warning(
            "Model %s checkpoint contains a non-weights-only NumPy scalar; "
            "using trusted full-object deserialization for %s.",
            config["model_id"],
            _relative_path(path),
        )
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not deserialize checkpoint for {config['model_id']!r}: "
                f"{_relative_path(path)}"
            ) from exc
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load checkpoint for {config['model_id']!r}: "
            f"{_relative_path(path)}"
        ) from exc


def _extract_state_dict(checkpoint, config: dict) -> dict:
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ModelLoadError(
            f"Checkpoint for {config['model_id']!r} is not a non-empty state "
            "dictionary or wrapper dictionary."
        )
    checkpoint_format = "raw state_dict"
    for key in ("model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            state = candidate
            checkpoint_format = f"wrapper[{key}]"
            break
    else:
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            state = checkpoint
        else:
            raise ModelLoadError(
                f"Checkpoint for {config['model_id']!r} has no supported "
                "'model_state_dict' or 'state_dict' entry."
            )

    if not state:
        raise ModelLoadError(
            f"Checkpoint for {config['model_id']!r} contains an empty state dictionary."
        )
    logger.info(
        "Checkpoint model=%s format=%s state_keys=%d",
        config["model_id"],
        checkpoint_format,
        len(state),
    )
    prefixed = [key.startswith("module.") for key in state]
    if all(prefixed):
        state = {key[len("module.") :]: value for key, value in state.items()}
        logger.info("Removed the verified 'module.' prefix from checkpoint keys.")
    elif any(prefixed):
        raise ModelLoadError(
            f"Checkpoint for {config['model_id']!r} mixes prefixed and unprefixed keys."
        )
    return state


def _validate_metadata(checkpoint: dict, config: dict) -> None:
    classes_key = config["class_metadata_key"]
    if classes_key in checkpoint:
        classes = checkpoint[classes_key]
        if list(classes) != list(CLASS_NAMES):
            raise ModelLoadError(
                f"Checkpoint for {config['model_id']!r} declares class order "
                f"{list(classes)}; expected {list(CLASS_NAMES)}."
            )
    image_size_key = config["image_size_metadata_key"]
    if image_size_key and image_size_key in checkpoint:
        image_size = tuple(checkpoint[image_size_key])
        expected = (config["input_size"], config["input_size"])
        if image_size != expected:
            raise ModelLoadError(
                f"Checkpoint for {config['model_id']!r} declares input size "
                f"{image_size}; expected {expected}."
            )


class Classifier:
    """A cached, device-placed classifier for the configured active model."""

    def __init__(self, model_config: dict | None = None):
        self.config = model_config or get_active_model_config()
        self.model_id = self.config["model_id"]
        self.checkpoint_path = require_checkpoint(self.config)
        self.device = resolve_device()
        self.model: nn.Module | None = None

        started = time.perf_counter()
        logger.info(
            "Loading model id=%s architecture=%s checkpoint=%s device=%s",
            self.model_id,
            self.config["architecture"],
            _relative_path(self.checkpoint_path),
            self.device,
        )
        try:
            model = build_architecture(
                self.config["architecture"], num_classes=len(CLASS_NAMES)
            )
            checkpoint = _load_checkpoint(self.checkpoint_path, self.config)
            _validate_metadata(checkpoint, self.config)
            state = _extract_state_dict(checkpoint, self.config)
            try:
                result = model.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                raise ModelLoadError(
                    f"Checkpoint for {self.model_id!r} is incompatible with its "
                    f"{self.config['architecture']!r} architecture. "
                    f"Strict load failed: {exc}"
                ) from exc
            model.to(self.device)
            model.eval()
        except ModelLoadError:
            raise
        except torch.cuda.OutOfMemoryError as exc:
            raise ModelLoadError(
                f"Out of memory while loading {self.model_id!r} on {self.device}. "
                "Use a smaller model or set MODEL_DEVICE='cpu'."
            ) from exc
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to initialize model {self.model_id!r} on {self.device}: {exc}"
            ) from exc

        self.model = model
        logger.info(
            "Loaded model id=%s parameters=%d loaded_keys=%d missing=%d "
            "unexpected=%d elapsed_ms=%.1f",
            self.model_id,
            sum(parameter.numel() for parameter in model.parameters()),
            len(state),
            len(result.missing_keys),
            len(result.unexpected_keys),
            (time.perf_counter() - started) * 1000,
        )

    @property
    def display_name(self) -> str:
        return self.config["display_name"]

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ModelLoadError(f"Model {self.model_id!r} is not loaded.")
        try:
            with torch.inference_mode():
                logits = self.model(batch)
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise ModelInferenceError(
                f"Out of memory during {self.model_id!r} inference on {self.device}. "
                "Reduce the batch size or use CPU."
            ) from exc
        except RuntimeError as exc:
            raise ModelInferenceError(
                f"Inference failed for model {self.model_id!r} on {self.device}: {exc}"
            ) from exc

        if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
            raise ModelInferenceError(
                f"Model {self.model_id!r} returned shape {tuple(logits.shape)}; "
                f"expected (batch, {len(CLASS_NAMES)})."
            )
        if not torch.isfinite(logits).all():
            raise ModelInferenceError(
                f"Model {self.model_id!r} returned non-finite logits."
            )
        return F.softmax(logits, dim=1)

    def predict(self, crops) -> list[dict]:
        if not crops:
            return []

        started = time.perf_counter()
        configured_batch = int(self.config["inference_batch_size"])
        batch_size = configured_batch or len(crops)
        probabilities_batches = []
        for start in range(0, len(crops), batch_size):
            chunk = crops[start : start + batch_size]
            try:
                batch = torch.cat(
                    [
                        _PREPROCESSORS[self.config["preprocessing"]](crop.image)
                        for crop in chunk
                    ],
                    dim=0,
                ).to(self.device)
            except (TypeError, ValueError, cv2.error) as exc:
                raise ModelInferenceError(
                    f"Failed to preprocess a crop for {self.model_id!r}: {exc}"
                ) from exc
            probabilities_batches.append(self._forward(batch).cpu().numpy())

        probabilities = np.concatenate(probabilities_batches, axis=0)
        results = []
        names = self.config["class_names"]
        for crop, probs in zip(crops, probabilities):
            index = int(np.argmax(probs))
            results.append(
                {
                    "crop_id": crop.crop_id,
                    "x1": crop.x1,
                    "y1": crop.y1,
                    "x2": crop.x2,
                    "y2": crop.y2,
                    "width": crop.x2 - crop.x1,
                    "height": crop.y2 - crop.y1,
                    "predicted_class": names[index],
                    "confidence": round(float(probs[index]), 4),
                    "probabilities": {
                        name: round(float(probs[i]), 4)
                        for i, name in enumerate(names)
                    },
                }
            )
        logger.info(
            "Inference model=%s input=%s output=%s crops=%d batch_size=%d "
            "device=%s elapsed_ms=%.1f",
            self.model_id,
            (
                len(crops),
                int(self.config["channels"]),
                int(self.config["input_size"]),
                int(self.config["input_size"]),
            ),
            (len(crops), len(CLASS_NAMES)),
            len(crops),
            batch_size,
            self.device,
            (time.perf_counter() - started) * 1000,
        )
        return results
