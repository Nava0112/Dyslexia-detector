# Dyslexia handwriting screening

Local FastAPI application for handwriting screening. It segments handwriting
regions, classifies each crop into `Corrected`, `Normal`, or `Reversal`, and
returns a research screening result plus an annotated image. It is not a
medical diagnostic system.

Three trained image classifiers can be selected. Model D v4 remains the
default and preserves the original API, segmentation, artifacts, and crop
classification behavior.

## Supported models

| Model ID | Display name | Architecture | Checkpoint |
| --- | --- | --- | --- |
| `model_d_v4` | Model D v4 | MobileNetV3-Small trunk, inception/SE refinement, CBAM fusion, SPP head | `models/model_d_v4_best.pth` |
| `vit_b16` | ViT-B/16 Dyslexia Classifier | torchvision ViT-B/16, 3-class head | `models/best_vit_b16_dyslexia.pth` |
| `mobilenet_v3_small` | MobileNetV3-Small Dyslexia Classifier | torchvision MobileNetV3-Small, grayscale first convolution, 3-class head | `models/model_a_mobilenet_v3_small.pth` |

All three checkpoints are **image-level three-class classifiers**, not
pixel-wise segmentation models. The existing segmentation code in
`backend/segmentation.py` supplies the crops.

### Verified preprocessing and checkpoint contracts

| Model | Stored weights | Verified input contract | Output |
| --- | --- | --- | --- |
| Model D v4 | `model_state_dict`; metadata `class_names`, `img_size` | 160x160 grayscale, square black padding, `INTER_AREA`, CLAHE 2.0/8x8, scale to `[-1, 1]` | 3 logits -> softmax |
| MobileNetV3-Small | `model_state_dict`; metadata `class_names` | 128x128 grayscale, square black padding, `INTER_AREA`, CLAHE 2.0/8x8, 3x3 Gaussian blur, adaptive threshold (`GAUSSIAN_C`, inverse, block 15, C 5), scale to `[0, 1]` | 3 logits -> softmax |
| ViT-B/16 | `model_state_dict`; metadata `classes`, `best_val_f1` | 224x224, square black padding, `INTER_AREA`, grayscale converted to RGB, ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` | 3 logits -> softmax |

The class order for every checkpoint is verified against the API order:
`("Corrected", "Normal", "Reversal")`. A checkpoint with a different declared
order is rejected at load time.

Architecture evidence:

- Model D v4: `Model_D_v4_DEBUGGED.ipynb` and the original production
  implementation.
- ViT-B/16: `models/VT2.ipynb` (training) and
  `models/Dyslexia_Handwriting_Inference_Segmentation_Fixed.ipynb` (inference
  preprocessing).
- MobileNetV3-Small: `models/Dyslexia_V2_60k_MobileNetV3_Small_GPU.ipynb`.

All checkpoint loads are `strict=True` against the selected architecture. The
loader does not silently drop missing/unexpected keys.

## Switching the active model

Change **only** `ACTIVE_MODEL` in `backend/config.py`, then restart the backend
process:

```python
# backend/config.py
ACTIVE_MODEL = "model_d_v4"          # default
# ACTIVE_MODEL = "vit_b16"
# ACTIVE_MODEL = "mobilenet_v3_small"
```

Do not edit `backend/model.py`, `backend/pipeline.py`, `app.py`, the frontend,
or `.env` to switch models. The former `DYSLEXIA_CHECKPOINT` environment
override is no longer used; model selection has one source of truth. Unknown
IDs fail at configuration import time with a list of supported IDs.

Device selection is also in `backend/config.py`:

```python
MODEL_DEVICE = "auto"   # CUDA when available, otherwise CPU
# MODEL_DEVICE = "cpu"
# MODEL_DEVICE = "cuda"  # unavailable CUDA is a clear error, not a fallback
```

## Required files

Checkpoints must remain in `models/` with these exact names:

```text
models/model_d_v4_best.pth
models/best_vit_b16_dyslexia.pth
models/model_a_mobilenet_v3_small.pth
```

`model_d_v4_best.pth.zip` in the repository root is the legacy copy used before
this integration. It is byte-identical to `models/model_d_v4_best.pth`
(SHA-256 `89DC49B7...E33C7C`) and is not selected by the new configuration.

## Install and run

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional improved line segmentation:
pip install -r requirements-optional.txt
python app.py
```

The API listens on `http://127.0.0.1:2000`. Endpoints:

- `GET /ready` validates the selected checkpoint and configured device.
- `POST /analyze` accepts JPG/PNG/WebP and returns the existing
  `AnalysisResponse` shape.
- `GET /artifacts/{job_id}/{filename}` serves `annotated.png` and
  `result.json`.

No new runtime package is required. `torchvision` was already a dependency and
provides both `vit_b_16` and `mobilenet_v3_small`.

## Tests and smoke checks

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite covers configuration validation, default selection, strict loading,
wrong-architecture rejection, missing checkpoints, preprocessing contracts,
crop bounds/alignment, repeated inference reuse, CPU inference, per-model
inference, and all three models end to end. It uses the real checkpoints and a
deterministic synthetic handwriting fixture. The exact Model D v4 regression
assertion is skipped when the optional Kraken package is unavailable because
line segmentation then uses the OpenCV fallback.

Reference fixture: 1300x900 image, 6 lines, 72 crops. Model D v4 returns
`Corrected=3`, `Normal=39`, `Reversal=30`, score `0.4458`, and an identical
annotated image before and after this integration.

## Measured smoke performance

Measured with all 72 crops from the same fixture, CPU-only, after a two-crop
warm-up. Inference excludes segmentation time.

| Model | Load | Inference (72 crops) | Per crop | RSS increase after load | Peak process RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model D v4 | 2.79 s | 0.68 s | 9.42 ms | 151.9 MB | 450.7 MB |
| MobileNetV3-Small | 2.11 s | 0.36 s | 4.93 ms | 130.8 MB | 404.9 MB |
| ViT-B/16 | 3.30 s | 20.69 s | 287.31 ms | 1105.8 MB | 1105.8 MB |

Hardware and environment: Windows 11 (10.0.26100), Intel Core i3-1115G4,
4 logical CPUs, 7.7 GB RAM, PyTorch `2.4.1+cpu`, torchvision `0.19.1+cpu`.
Segmentation took 3.4-5.6 s per image in these runs and is shared by all models.
CUDA was unavailable, so no GPU latency or memory numbers are claimed.

## Validation limitations

- No labeled validation dataset is included in this repository. Accuracy, IoU,
  Dice, precision, recall, and crop-quality comparisons were therefore **not**
  re-measured here.
- Values stored in checkpoints (`best_val_accuracy` for Model D v4 and
  MobileNetV3-Small, `best_val_f1` for ViT-B/16) come from different training
  notebooks, splits, and task definitions. They are metadata, not a
  like-for-like benchmark, and must not be used to claim one model is superior.
- The external notebooks evaluate character-level crops. This application keeps
  its existing connected-component/line crop segmentation and applies each
  classifier to those crops. Segmentation quality and crop definitions
  therefore remain a shared limitation and can affect every model's result.
- ViT-B/16 uses substantially more memory and CPU time than the two CNN models.
  Its checkpoint contains a NumPy scalar (`best_val_f1`), which PyTorch 2.4
  cannot load in `weights_only=True` mode. The configured, repository-local ViT
  checkpoint is therefore explicitly trusted for full deserialization; all
  other checkpoints use `weights_only=True`. Do not point this loader at an
  untrusted ViT file.
- CUDA behavior is implemented and covered by a skipped-if-unavailable test, but
  has not been run on a CUDA machine here.

## Troubleshooting

- **Unknown model ID**: `ConfigurationError` at import lists the supported IDs.
- **Missing checkpoint**: `/ready` and `/analyze` return HTTP 503 naming the
  selected model and configured checkpoint. Check the exact filename in
  `models/`.
- **Incompatible state dictionary**: the server logs the selected ID,
  architecture, checkpoint, device, and the strict-load mismatch. The app does
  not fall back to Model D v4 or another model.
- **CUDA selected but unavailable**: startup/load fails with an actionable
  message. Set `MODEL_DEVICE = "auto"` or `"cpu"`.
- **Out of memory with ViT**: use CPU or a machine with enough memory; the
  configured ViT batch size is 8 crops and can be lowered in
  `backend/config.py`.

## Optional LLM guidance

The analysis view can request plain-language learning guidance from OpenRouter
after a screening completes. Copy `.env.example` to `.env` and set
`OPENROUTER_API_KEY`. `OPENROUTER_MODEL` may be set to any model available
through OpenRouter.

Only the status, score, threshold, line/crop counts, and the three numerical
class counts are sent to the guidance service. The uploaded handwriting image
is never sent. If OpenRouter is unavailable or not configured, the primary
analysis results remain available and the guidance panel shows a retry state.
