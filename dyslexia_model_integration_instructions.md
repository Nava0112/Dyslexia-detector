# Integration Instructions: Multiple Dyslexia Segmentation Models

## 1. Goal and non-negotiable requirements

Integrate the two additional trained models into the existing
`D:\dyslexia` application so that the model used for inference can be
changed **only by editing `backend/config.py`**.

The project currently runs with **Model D v4**. Preserve its current
behavior as the default and as the fallback configuration. Adding the
new models must not silently change Model D v4 preprocessing,
segmentation, output formats, API responses, artifact generation, or
frontend behavior.

The repository currently contains:

``` text
D:\dyslexia\
├── app.py
├── backend\
│   ├── config.py
│   ├── model.py
│   ├── pipeline.py
│   ├── segmentation.py
│   ├── schemas.py
│   ├── guidance.py
│   └── llm_config.py
├── frontend\
├── artifacts\
├── requirements.txt
├── requirements-optional.txt
└── models\                  # trained model assets
```

The repository also contains notebooks and model assets with names shown
in the supplied screenshot. Visible names include:

-   `best_vit_b16_dyslexia.pth`
-   `model_a_mobilenet_v3_small.pth`
-   `model_d_v4_best.pth`
-   `Dyslexia_Handwriting_Inference_Segmen...`
-   `Dyslexia_V2_60k_MobileNetV3_Small_GPU...`
-   `VT2`

Some screenshot names are truncated. Inspect the actual files and
extensions before implementing anything. Do not infer the architecture
from a filename alone.

## 2. First inspect the repository; do not code from assumptions

Before editing, inspect these files and record how the current
implementation works:

1.  `backend/config.py` --- current model path, device settings, image
    settings, thresholds, and other model-related constants.
2.  `backend/model.py` --- model construction, checkpoint loading,
    device placement, and inference helpers.
3.  `backend/pipeline.py` --- end-to-end processing order and where the
    model is called.
4.  `backend/segmentation.py` --- masks, class IDs, connected
    components, crop generation, and post-processing.
5.  `backend/schemas.py` and `app.py` --- request/response contracts and
    error handling.
6.  `requirements.txt` and `requirements-optional.txt` --- installed
    model-library dependencies.
7.  The notebooks and Python source that define each candidate model's
    architecture and inference preprocessing.
8.  The contents and metadata of every checkpoint in `models/`.

Use the notebooks as evidence for the architecture and
training/inference conventions. Do not copy an entire notebook into
production code. Extract the minimal, reusable architecture definition
and the exact inference-time preprocessing needed for that model.

For each checkpoint, determine and document:

-   Its intended architecture and matching source file/notebook.
-   Whether it is a raw `state_dict`, a dictionary containing
    `state_dict`/`model_state_dict`, or a serialized full model.
-   The expected input tensor shape, channel count, color order,
    normalization, resizing, and padding behavior.
-   The output tensor shape and meaning of each output channel/class.
-   Whether the output is logits, probabilities, or already-decoded
    labels.
-   The class-to-color / class-to-label mapping.
-   Any custom layers, model-library versions, and special loading
    requirements.
-   Whether the checkpoint loads strictly into the proposed
    architecture.
-   Whether it expects image-level classification, semantic
    segmentation, instance segmentation, or another task. Do not
    integrate a classifier as if it were a pixel-wise segmenter.

### Likely model mapping to verify

Treat the following as *hypotheses to verify*, not facts:

  --------------------------------------------------------------------------------------------------------------
  Candidate               Checkpoint                                Architecture source to locate
  ----------------------- ----------------------------------------- --------------------------------------------
  Current Model D v4      `models/model_d_v4_best.pth`              Existing production implementation and/or
                                                                    `Model_D_v4_DEBUGGED.ipynb`

  Candidate ViT-B/16      `models/best_vit_b16_dyslexia.pth`        The actual notebook/source matching
                                                                    `Dyslexia_Handwriting_Inference_Segmen...`
                                                                    or `VT2`

  Candidate MobileNetV3   `models/model_a_mobilenet_v3_small.pth`   The actual notebook/source matching
  Small                                                             `Dyslexia_V2_60k_MobileNetV3_Small_GPU...`
  --------------------------------------------------------------------------------------------------------------

If inspection shows a different mapping, follow the checkpoint metadata
and architecture implementation, and update the table and documentation
accordingly.

## 3. Configuration must be the single switching point

Add a clear, validated model selector in `backend/config.py`. The user
must not need to edit `backend/model.py`, `backend/pipeline.py`,
`app.py`, the frontend, or `.env` to switch models.

Prefer a small registry keyed by stable model IDs, plus one active-model
constant. Use the repository's existing naming conventions if they are
already established. For example, the configuration could express the
following conceptual structure:

``` python
# backend/config.py — conceptual example; adapt to existing conventions.
ACTIVE_MODEL = "model_d_v4"

MODEL_CONFIGS = {
    "model_d_v4": {
        "display_name": "Model D v4",
        "checkpoint_path": MODEL_DIR / "model_d_v4_best.pth",
        "architecture": "model_d_v4",
    },
    "vit_b16": {
        "display_name": "ViT-B/16 Dyslexia Model",
        "checkpoint_path": MODEL_DIR / "best_vit_b16_dyslexia.pth",
        "architecture": "vit_b16",
    },
    "mobilenet_v3_small": {
        "display_name": "MobileNetV3 Small Dyslexia Model",
        "checkpoint_path": MODEL_DIR / "model_a_mobilenet_v3_small.pth",
        "architecture": "mobilenet_v3_small",
    },
}
```

This is an illustration, not code to paste blindly. In particular:

-   Use the real project path convention. If `config.py` currently uses
    `Path`, keep it consistent and resolve paths robustly relative to
    the repository root rather than the process's current working
    directory.
-   Verify the exact architecture identifiers and checkpoint filenames.
-   Preserve all existing Model D v4 constants and behavior.
-   Keep model selection separate from unrelated settings such as LLM
    configuration.
-   Do not move the selector to `.env`; the requirement is to switch it
    through `backend/config.py`.
-   Avoid duplicate or contradictory active-model variables.
-   Validate the active ID during startup. If it is unknown, raise a
    clear configuration error listing supported IDs rather than silently
    selecting a different model.
-   Validate the selected checkpoint exists before inference. Report the
    missing path and selected model clearly.
-   Do not add a frontend model selector unless explicitly requested.
    The active model is chosen in backend configuration.

A user should be able to switch models by changing only the active ID,
for example:

``` python
ACTIVE_MODEL = "model_d_v4"
# or "vit_b16"
# or "mobilenet_v3_small"
```

## 4. Separate architecture construction from checkpoint loading

Refactor only as much as needed to support the models without breaking
the existing code.

Recommended responsibilities:

-   `backend/config.py`: model registry, active model ID, paths, device
    policy, and model-specific inference settings.
-   `backend/model.py`: architecture factory, checkpoint loader, cached
    model instance, device placement, and common inference entry point.
-   `backend/pipeline.py`: orchestrate the existing pipeline using the
    common model inference interface.
-   `backend/segmentation.py`: retain common segmentation/crop logic
    where the output semantics are genuinely shared; isolate
    model-specific output decoding where they differ.
-   A new module such as `backend/model_architectures.py` or a small
    `backend/models/` Python package may contain architecture
    definitions/adapters if needed. Avoid circular imports and avoid
    confusing the Python source package with the trained-weight
    directory.

Create a factory or equivalent dispatch layer so the rest of the
application does not need architecture-specific `if` statements
scattered throughout the code. Conceptually:

1.  Read `ACTIVE_MODEL` from `backend/config.py`.
2.  Look up the selected model's configuration.
3.  Construct the matching architecture using the verified constructor
    arguments.
4.  Load the matching checkpoint.
5.  Move the model to the chosen device.
6.  Set evaluation mode.
7.  Apply that model's exact inference preprocessing.
8.  Run inference with gradients disabled.
9.  Decode the model's output into a documented common representation.
10. Pass the normalized result into the existing downstream pipeline.

Do not instantiate a new model for every request. Load it once per
process, cache the instance, and ensure initialization is safe if the
web server can process concurrent requests.

## 5. Checkpoint loading must be explicit and safe

Implement checkpoint loading based on the actual checkpoint format. Do
not assume all `.pth` files have the same structure.

-   For a raw `state_dict`, load the mapping directly.
-   For a wrapper dictionary, identify the actual weight key
    (`state_dict`, `model_state_dict`, or the verified key).
-   If keys contain a known prefix such as `module.`, remove it only
    when inspection confirms it is present and doing so is appropriate.
-   Do not silently use `strict=False` to hide architecture mismatches.
    Prefer strict loading. If non-strict loading is genuinely necessary,
    document each missing/unexpected key and explain why it is safe.
-   Never load a checkpoint into the wrong architecture merely because
    some tensor shapes happen to match.
-   Prefer loading weights with the safest supported PyTorch options for
    the checkpoint format. Only use full-object deserialization when the
    file genuinely requires it and its source is trusted.
-   Report incompatible tensor shapes, missing keys, unexpected keys,
    unsupported checkpoint formats, and missing files with actionable
    messages.
-   Keep the original checkpoint files unchanged.

If the notebook saves metadata such as class names, image size, or
architecture parameters, use and validate that metadata. If metadata is
absent, document the verified values from the training notebook.

## 6. Model-specific preprocessing and output adapters

Do not assume the three models share preprocessing or output semantics.
Each model needs a small adapter/configuration that specifies its
verified inference contract.

For each model, confirm and implement:

-   Image decode and channel order (RGB/BGR/grayscale).
-   Input size and whether aspect ratio is preserved.
-   Resize/interpolation method.
-   Padding strategy and pad value, if any.
-   Input range (`0–1`, `0–255`, or another range).
-   Normalization mean and standard deviation.
-   Tensor shape and batch dimension.
-   Device and dtype.
-   Output tensor shape.
-   Whether to apply `sigmoid`, `softmax`, `argmax`, or another decoder.
-   Class order and meaning of background/foreground classes.
-   Thresholds and morphology/post-processing settings.
-   How to reverse resize/padding so the output aligns with the original
    image.

Never apply `softmax` or `sigmoid` without confirming the
training/output contract. Do not assume output channel 0 means
background or that class IDs are interchangeable between models.

Convert each model's decoded output to a shared internal representation
only after preserving its semantics. If the existing pipeline expects a
particular mask type, shape, and class-ID convention, create an explicit
conversion step. Keep model-specific thresholds in that model's
configuration rather than changing global thresholds in a way that
alters Model D v4.

For crop extraction, preserve original-image coordinates. Test that
crops stay within image bounds, are not empty, and align with the source
image after all resizing and padding are reversed.

## 7. Preserve the current API and frontend contract

The frontend should not need model-specific logic. Keep existing request
fields, response JSON schema, artifact layout, and error behavior
compatible unless there is a verified bug that must be addressed
separately.

-   Keep the existing `result.json` fields and their meanings stable.
-   Keep the expected annotated-image path and crop/image naming stable.
-   Ensure `app.py` and `backend/pipeline.py` call the shared inference
    interface rather than loading checkpoints independently.
-   Do not add checkpoint tensors, internal model objects, or large raw
    arrays to API responses.
-   If useful for debugging, include the selected model ID in server
    logs or a backward-compatible metadata field only if it will not
    break current consumers.
-   Do not expose absolute local paths or sensitive server details to
    the frontend.
-   Keep current LLM guidance behavior independent of the segmentation
    model unless the existing design explicitly couples them.

## 8. Device, performance, and memory handling

-   Respect the project's existing device configuration. Use CUDA only
    when available and supported; otherwise use CPU if the application
    currently supports CPU operation.
-   Do not assume a model named `...GPU` can only run on a GPU; inspect
    its implementation and dependencies. Conversely, do not assume a
    model fits in GPU memory because it loaded on one machine.
-   Load the model once, call `.eval()`, and use
    `torch.inference_mode()` or the established no-gradient inference
    pattern.
-   Use an explicit, consistent device for both model and input tensors.
-   Handle unavailable CUDA, unsupported operators, and out-of-memory
    errors clearly. Do not silently change to a different model on
    failure.
-   Avoid repeatedly moving model weights between devices per request.
-   Measure initialization time, inference latency, and peak memory for
    each model using the same representative inputs. Record the hardware
    and image sizes used.
-   If optional dependencies are needed for a candidate model, document
    them and update the appropriate requirements file without breaking
    the current installation.

## 9. Logging and error handling

Log enough information to diagnose a failed load or inference:

-   selected model ID and architecture identifier;
-   checkpoint path (server-side logs only);
-   device selected;
-   checkpoint format and strict-load result;
-   input shape and output shape;
-   elapsed inference time;
-   clear exception context for missing dependencies or invalid output.

Do not log image contents, secrets, tokens, or unnecessary personal
data. Fail fast at startup for a bad model configuration or missing
checkpoint rather than waiting for a user request, if that fits the
application's current startup pattern.

Most importantly, **never silently fall back to Model D v4** when
another model is selected but fails to load. That would make model
comparisons unreliable. Fail with a clear error. Model D v4 should
remain the default only because `ACTIVE_MODEL` explicitly selects it.

## 10. Testing plan

Add tests using small fixtures or representative test images. Keep tests
deterministic where possible.

### Configuration and loading

-   The default active model is Model D v4.
-   Each supported model ID resolves to the correct architecture and
    checkpoint path.
-   An unknown model ID produces a clear error.
-   A missing checkpoint produces a clear error.
-   A checkpoint from one architecture is rejected by another
    architecture's loader.
-   Each checkpoint loads with the intended key mapping and expected
    strictness.
-   Repeated inference requests reuse the loaded model rather than
    reloading weights.

### Inference contract

For every model:

-   Valid image produces an output with the expected dimensions and
    type.
-   Output class IDs are within the documented range.
-   Output is not unexpectedly all background, all foreground, NaN, or
    infinite.
-   Resizing/padding is correctly reversed.
-   Inference runs on the configured device.
-   CPU inference works if CPU support is required.
-   CUDA inference works if CUDA is available and supported.
-   Empty, corrupt, grayscale, RGB, and unusually sized inputs are
    handled according to the existing API contract.

### End-to-end regression

Run the same representative image set through the existing Model D v4
configuration before and after the integration. Compare:

-   segmentation mask dimensions and class-ID meaning;
-   crop count and crop coordinates;
-   annotated image output;
-   `result.json` structure and field meanings;
-   API status codes and response shape;
-   generated artifact directories and filenames.

Model D v4 outputs should remain equivalent within a documented
tolerance. Avoid requiring byte-for-byte image equality if
nondeterministic operations or library changes make that unrealistic;
explain any difference.

### Model quality checks

Run each candidate against a fixed validation set that was not used
merely to tune its thresholds. Record the same metrics for all models,
using the project's actual task definition. For segmentation,
appropriate measures may include per-class IoU/Dice, precision/recall,
and crop quality, but choose metrics consistent with the labels and
evaluation setup. Also compare inference latency and memory. Do not
claim one model is better based on one sample or on its filename.

## 11. Suggested implementation sequence

1.  **Inventory** the actual files, full filenames, notebook
    architecture definitions, checkpoint formats, class mappings, and
    input/output contracts.
2.  **Capture baseline** outputs from the current Model D v4 pipeline
    for a small fixed regression set.
3.  **Formalize config** in `backend/config.py`, preserving Model D v4
    as the default.
4.  **Isolate current loader** behind a common factory/interface without
    changing Model D v4 behavior.
5.  **Integrate one candidate at a time**: architecture definition,
    strict checkpoint load, preprocessing, output decoder, then smoke
    test.
6.  Integrate the second candidate in the same way.
7.  **Normalize outputs** into the existing pipeline's expected
    representation without erasing model-specific class semantics.
8.  **Run unit and end-to-end tests** for every configured model.
9.  **Compare quality/performance** using the same fixed dataset and
    hardware.
10. **Update documentation and dependencies** only after the tested
    implementation is known.
11. Verify switching models requires changing only `ACTIVE_MODEL` in
    `backend/config.py` and restarting the backend process.

## 12. README and developer documentation

Document:

-   supported model IDs and their verified architecture names;
-   exact checkpoint filename for each model;
-   the single configuration line used to select the active model;
-   required model files and where they belong;
-   model-specific preprocessing/input size/class mappings;
-   supported devices and dependencies;
-   how to run smoke tests and the validation set;
-   known limitations and measured quality/performance;
-   how to diagnose missing checkpoints and incompatible state
    dictionaries.

Do not include claims about accuracy or superiority unless supported by
measured results.

## 13. Acceptance criteria

The integration is complete only when all of the following are true:

-   [ ] The real architecture corresponding to each checkpoint has been
    verified from source/notebook and checkpoint inspection.
-   [ ] Model D v4 remains the default.
-   [ ] The active model is selected only in `backend/config.py`.
-   [ ] Each model loads its own correct checkpoint and rejects
    incompatible weights.
-   [ ] Each model uses its own verified preprocessing and output
    decoding.
-   [ ] The common pipeline receives a consistent, documented output
    representation.
-   [ ] The API response schema and frontend behavior remain compatible.
-   [ ] Model D v4 regression tests pass.
-   [ ] Each new model passes checkpoint-load, inference, and end-to-end
    smoke tests.
-   [ ] CPU/GPU behavior and dependency requirements are documented and
    tested where applicable.
-   [ ] No silent fallback to another model occurs after a load or
    inference failure.
-   [ ] README instructions are accurate and model comparisons use
    measured validation results.

## Final implementation constraint

Make the smallest maintainable changes necessary. Do not rewrite the
entire application, retrain the models, rename or overwrite checkpoint
files, modify `.env` for model selection, or alter the frontend just to
support backend model switching. Inspect first, implement one model at a
time, and preserve the existing Model D v4 path as the regression
baseline.
