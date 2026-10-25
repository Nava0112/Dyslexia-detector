# Implementation Guide: Improve Letter Segmentation in the Dyslexia Screening Pipeline

## Goal

Improve the current image-to-character-crop stage so that new paragraph
images produce more reliable character crops. The observed failure
includes overlapping boxes, boxes spanning multiple letters, fragmented
letters, and inconsistent crop dimensions.

**Do not replace the classifier or redesign the entire application
first.** Diagnose and improve `extract_crops()` in `segmentation.py`,
measure the change on representative images, and only introduce a
learned character detector if classical computer-vision methods remain
inadequate.

This is an engineering guide, not a claim that the resulting screening
score diagnoses dyslexia. The screening logic and threshold require
independent validation.

------------------------------------------------------------------------

## 1. Preserve the current integration contract

The current pipeline calls:

``` python
crops, line_boxes, _ = extract_crops(image)
predictions = self.classifier.predict(crops)
```

Keep this interface compatible unless there is a deliberate, coordinated
change to `AnalysisPipeline`.

The implementation should continue to provide:

-   `crops`: character candidate images in reading order, prepared in
    the same format expected by the trained classifier.
-   `line_boxes`: detected text-line boxes in the coordinate system of
    the original input image.
-   A third value for debugging metadata, if that is what the existing
    function currently returns. Inspect callers before changing its
    meaning.

The classifier's prediction rows are expected by `AnalysisPipeline` to
include `predicted_class`, `x1`, `y1`, `x2`, and `y2`. Preserve this
structure or update every consumer consistently.

### First task: inspect the existing implementation

Before editing, document:

1.  The preprocessing sequence.
2.  How text lines are detected.
3.  How candidate regions are generated.
4.  Which filters remove candidates.
5.  Whether components are merged or split.
6.  How coordinates are transformed back to the original image.
7.  How crops are padded, resized, and normalized.
8.  Whether the classifier's training pipeline applies the same
    preprocessing.

Do not stack new morphological operations onto the current pipeline
without identifying what each existing operation does. Dilation and
closing can merge letters; opening and erosion can erase thin strokes.

------------------------------------------------------------------------

## 2. Add a segmentation debug mode first

Make the pipeline save intermediate images for a few test inputs. Debug
artifacts should be optional and should not alter normal inference.

Save at least:

1.  Original input.
2.  Grayscale image.
3.  Contrast-normalized image, if used.
4.  Binary image before morphology.
5.  Binary image after each morphology operation.
6.  Detected text-line regions.
7.  Connected-component visualization, with component IDs or boxes.
8.  Candidate boxes before filtering.
9.  Candidate boxes after filtering and grouping.
10. Final crops arranged in reading order.

For each candidate, record metadata such as:

-   Bounding box `(x, y, width, height)`.
-   Pixel area.
-   Text-line ID.
-   Component IDs grouped into that candidate.
-   Whether the candidate was merged, split, filtered, or retained.
-   A reason for rejecting a candidate.

Use a separate debug output directory or a configurable debug flag.
Avoid creating unlimited artifacts for every production request.

**Decision rule:** inspect the binary image before changing anything
else. If the strokes are already missing or joined together there, fix
preprocessing. If the binary text looks reasonable but the boxes are
wrong, focus on grouping and segmentation.

------------------------------------------------------------------------

## 3. Make preprocessing robust to new images

Do not assume all inputs share the same resolution, font size, contrast,
background, or lighting.

### 3.1 Validate input

-   Check that the image is non-empty and has a supported shape.
-   Handle grayscale and color inputs consistently.
-   Reject invalid inputs with a clear error rather than silently
    returning misleading crops.
-   Keep a copy of the original image for final annotation.

### 3.2 Normalize scale carefully

-   Estimate text height or line height.
-   If the text is too small, upscale it before segmentation.
-   Prefer scale decisions based on estimated text size rather than one
    fixed resize factor for every image.
-   Avoid downscaling already-small text.
-   Track any scale transform so detected boxes can be mapped back to
    original coordinates.

Upscaling can help algorithms operate on small strokes, but it cannot
restore information that was never captured.

### 3.3 Compare thresholding methods

Evaluate at least:

-   Otsu global thresholding for reasonably uniform backgrounds.
-   Adaptive Gaussian thresholding for uneven illumination.
-   A contrast-normalization step where the background is uneven.

Do not automatically combine every method. Select the approach that
works best on a validation set, or use a small, explicit fallback
strategy when image-quality checks indicate a failure.

Ensure the foreground/background polarity is consistent with later
operations.

### 3.4 Use morphology conservatively

-   Use opening only when it removes noise without erasing thin
    character strokes.
-   Use closing only when it repairs broken strokes without joining
    neighboring characters.
-   Avoid large structuring elements relative to the character size.
-   Inspect each operation's output independently.

For reversal-related screening, never apply horizontal flipping,
vertical flipping, or rotation as generic preprocessing. Such
transformations can change the feature being studied and contaminate
labels.

------------------------------------------------------------------------

## 4. Detect text lines before individual characters

The supplied example has visible line structure. Use that structure to
constrain segmentation.

Recommended sequence:

1.  Estimate the foreground text mask.
2.  Detect candidate line regions using horizontal projection, connected
    regions, or a line-detection method suitable for the image.
3.  Sort lines from top to bottom.
4.  Crop each line with a small margin.
5.  Estimate each line's height and baseline where practical.
6.  Segment characters within each line rather than across the entire
    page.

Horizontal projection profiles count foreground pixels in each row.
Low-ink horizontal bands can suggest gaps between lines, but ascenders,
descenders, and closely spaced lines can complicate boundaries. Add
margins and avoid cutting off strokes.

If lines are tilted, estimate and correct the line angle carefully. Keep
track of coordinate transforms so output boxes are still expressed in
the original image coordinates.

Do not use a line detector's boxes as character boxes. They represent
different levels of segmentation.

------------------------------------------------------------------------

## 5. Generate and group candidate character regions

### 5.1 Start with connected components as candidates

Connected-component labeling is a useful candidate generator for clean
printed text, but a connected component is not necessarily one
character.

It may represent:

-   One whole character.
-   A dot or accent.
-   A fragment of a broken character.
-   Several touching characters.
-   Punctuation or noise.

Treat components as candidates, not guaranteed characters.

### 5.2 Use relative geometric features

Avoid hard-coded pixel cutoffs that only work for the original sample.
Estimate a typical text-line height and use relative features,
including:

-   Component height relative to line height.
-   Width-to-height ratio.
-   Area relative to line height squared.
-   Vertical position relative to the line baseline.
-   Horizontal distance to nearby components.
-   Overlap in vertical range.
-   Alignment with the line.

Use these features to flag likely noise and unusual candidates, not to
reject every unusual shape. Narrow letters such as `i` and `l`,
punctuation, and dots naturally differ from average character
dimensions.

### 5.3 Group fragments conservatively

A letter such as `i` may consist of a dot and a stem. A broken stroke
can also create multiple components.

Group components only when their geometry supports a shared character:

-   They are in the same text line.
-   Their horizontal separation is small relative to typical character
    width.
-   Their vertical positions are compatible with a known pattern, such
    as a dot above a stem.
-   Their combined bounding box remains plausible for one character.

Keep the original component IDs and grouping decisions in debug
metadata. Avoid merging solely because two components are close: this
can combine adjacent letters.

### 5.4 Detect possible merged characters

Flag unusually wide candidates relative to the typical character width
for that line. Do not split every wide candidate automatically; letters
such as `m` and `w` are naturally wider than `i` or `l`.

For flagged candidates, compare these options:

1.  Vertical projection profile within the candidate.
2.  Candidate split locations that pass through low-ink valleys.
3.  Distance transform and watershed when the shapes provide useful
    separation cues.
4.  A character detector or instance-segmentation model if the classical
    methods cannot resolve the boundary.

Projection profiles are most helpful when adjacent characters have a
visible valley. Watershed can over-segment a single character, so it
should be applied selectively and its output validated.

Do not infer character boundaries from word spaces alone. Words contain
multiple characters, and kerning makes character gaps inconsistent.

### 5.5 Prevent duplicate and overlapping boxes

The screenshot shows overlapping rectangles. Add checks before accepting
final candidates:

-   Compare boxes within the same line.
-   Detect duplicate or near-identical boxes.
-   Identify large overlaps and candidates whose boxes contain one
    another.
-   Check whether two candidates came from the same component pixels.
-   Resolve conflicts using the segmentation evidence, not only a
    generic overlap threshold.

Non-maximum suppression is useful for duplicate predictions from an
object detector, but it is not a complete solution for
connected-component segmentation. Two valid adjacent characters may have
overlapping bounding boxes, and a single wide box may be an incorrect
merge rather than a duplicate.

Do not blindly remove every overlapping box. Record the reason for each
removal and inspect the debug visualization.

------------------------------------------------------------------------

## 6. Normalize crops to match classifier training

A segmentation improvement can still lower accuracy if the resulting
crops differ from the training inputs.

For every accepted crop:

1.  Extract pixels from the appropriate source image.
2.  Include a small margin so the character's strokes are not clipped.
3.  Preserve the character's aspect ratio when resizing.
4.  Place it on the same canvas dimensions used during training.
5.  Match foreground/background polarity.
6.  Match grayscale or color channels.
7.  Match numeric scaling and normalization.
8.  Use the same interpolation and padding policy as training when
    possible.

Do not stretch all crops independently to a square if that distorts
character shapes. Resize proportionally, then pad to the expected
dimensions.

Keep reversal-sensitive visual information intact. Do not flip or rotate
characters to make them look more standardized. Do not apply any
transformation that could turn a reversed letter into a corrected one.

Add tests that compare the preprocessing output for a sample training
image against the expected training-time representation.

------------------------------------------------------------------------

## 7. Validate each candidate before classification

For each crop, validate:

-   The bounding box lies within the image.
-   Width and height are positive.
-   The crop is non-empty.
-   The crop is not mostly background.
-   The dimensions are plausible relative to its text line.
-   The crop has enough margin to avoid clipping.
-   It is not a duplicate of another candidate from the same pixels.

Use relative thresholds and preserve unusual but valid letters. A fixed
minimum area alone can accidentally remove dots, punctuation, or narrow
characters.

If a line produces implausibly few or many candidates, mark the
segmentation as low-confidence and save diagnostics. Do not silently
treat every crop as reliable.

Consider returning internal debug metadata such as:

-   Candidate count before and after filtering.
-   Number of merged or fragmented candidates flagged.
-   Number of rejected candidates and rejection reasons.
-   Per-line candidate counts.
-   Warnings about unusually wide boxes or empty crops.

Keep the public return structure compatible with current callers, or
update all callers together.

------------------------------------------------------------------------

## 8. Evaluate alternatives only after establishing a baseline

If classical segmentation still fails on touching or stylized
characters, compare a learned method against the improved baseline.

### Option A: Character-level object detector

Use a detector trained to return a bounding box for each character. This
can be a practical next step if you can annotate character boxes for
representative images.

Requirements:

-   Training images similar to expected user inputs.
-   Character-level bounding-box annotations.
-   Consistent class/label definitions.
-   Evaluation on images not used for training.
-   Careful handling of overlapping or duplicate predictions.

A generic text detector usually finds words or text lines, not
individual letters. Verify that the selected model actually supports
character-level detection.

### Option B: Instance segmentation

Use a model that predicts a separate mask for each character when
precise pixel regions are important. This requires instance-level masks
or a suitable pretrained model and may be more expensive to train and
run.

### Option C: OCR as a supporting tool

OCR can help locate text lines or words and can be useful for
diagnostics. Standard OCR output does not guarantee individual character
image crops. Do not assume OCR word boxes solve the segmentation
problem.

### Recommendation

Keep the existing OpenCV method if it can be made reliable on the target
input distribution. Introduce a learned character detector only if
measured merge and fragmentation errors remain unacceptable after the
simpler fixes.

------------------------------------------------------------------------

## 9. Build a representative evaluation set

Create a fixed collection of images covering:

-   The current working examples.
-   The failing example in the screenshot.
-   Different text sizes and resolutions.
-   Different line spacing and paragraph lengths.
-   Light and dark backgrounds.
-   Photographs with shadows or skew.
-   Different fonts, including the stylized or connected font that is
    currently failing.
-   Any handwriting the application is expected to support.

Annotate ground-truth character boxes for a manageable subset. If the
goal is to detect character instances, evaluate against character
instances rather than comparing only total crop counts.

Track these metrics:

### Detection recall

Fraction of ground-truth characters matched by a predicted crop.

### Detection precision

Fraction of predicted crops that match a real character.

### Merge rate

Fraction of predicted crops that contain multiple ground-truth
characters.

### Fragmentation rate

Fraction of ground-truth characters represented by multiple predicted
crops.

### Box quality

Use intersection over union (IoU) between predicted and ground-truth
boxes, with a clearly specified matching rule.

### Downstream classification

Evaluate classifier performance on correctly segmented crops separately
from end-to-end performance. Otherwise, it is difficult to tell whether
an error originates in segmentation or classification.

Review metrics by image type and font. An overall average can hide
severe failures on one type of input.

------------------------------------------------------------------------

## 10. Integrate changes safely with `AnalysisPipeline`

The current pipeline assumes the number of predictions is the number of
usable crops:

``` python
predictions = self.classifier.predict(crops)
usable = len(predictions)
```

After changing segmentation:

-   Confirm the classifier returns one prediction per crop.
-   Confirm each prediction's box corresponds to the crop it classified.
-   Confirm coordinates refer to the original image, not a resized or
    line-cropped image.
-   Ensure line boxes and crop boxes use the same coordinate convention.
-   Keep class names and `CLASS_COLORS` consistent.
-   Handle empty crop lists without calling the classifier if it cannot
    accept them.
-   Keep debug output optional.
-   Test artifact saving and error handling.
-   Confirm insufficient segmentation does not produce a misleading
    screening result.

The current screening score is calculated from predicted class counts.
Better segmentation changes those counts, so the score and threshold
behavior must be re-evaluated after segmentation changes. The score
should not be described as a calibrated probability unless validation
supports that interpretation.

------------------------------------------------------------------------

## 11. Suggested implementation phases

### Phase 1 --- Diagnose

-   [ ] Inspect `segmentation.py` and all helper functions.
-   [ ] Save intermediate images.
-   [ ] Reproduce the screenshot's failure.
-   [ ] Determine whether the binary mask, line detection, component
    grouping, or crop filtering causes the problem.
-   [ ] Record baseline precision, recall, merge rate, and fragmentation
    rate on a small annotated set.

### Phase 2 --- Improve the existing algorithm

-   [ ] Make preprocessing scale-aware.
-   [ ] Compare global and adaptive thresholding.
-   [ ] Detect text lines before segmenting characters.
-   [ ] Replace fixed pixel thresholds with relative geometric rules
    where appropriate.
-   [ ] Group disconnected components conservatively.
-   [ ] Split only likely merged candidates.
-   [ ] Detect duplicate and suspiciously overlapping boxes.
-   [ ] Normalize crops to match training.

### Phase 3 --- Verify end-to-end behavior

-   [ ] Visualize final boxes and all crops in reading order.
-   [ ] Test different resolutions and text styles.
-   [ ] Check crop-to-prediction correspondence.
-   [ ] Verify original-image coordinates.
-   [ ] Compare classification performance before and after the change.
-   [ ] Revalidate the screening threshold if the distribution of
    predicted classes changes.

### Phase 4 --- Decide whether to replace the method

-   [ ] Identify the failures that remain after Phase 2.
-   [ ] Compare a character detector or instance-segmentation approach
    on the same test set.
-   [ ] Compare quality, latency, annotation effort, and deployment
    complexity.
-   [ ] Keep the simplest method that meets the project's measured
    requirements.

------------------------------------------------------------------------

## 12. Definition of done

Do not consider the work complete merely because the boxes look better
on one screenshot. The implementation is ready for further integration
when:

-   Boxes rarely contain multiple characters on the representative test
    set.
-   Fragmented characters are substantially reduced.
-   Valid narrow characters and dots are retained.
-   Crops do not clip character strokes.
-   Output order and coordinates are correct.
-   The classifier receives crops with the expected input format.
-   End-to-end metrics improve on held-out images.
-   Failure cases can be inspected using debug artifacts.
-   Reversal-sensitive transformations are not introduced.
-   The system reports insufficient or low-confidence input rather than
    presenting an unreliable result as conclusive.

## Final recommendation

Start by instrumenting and inspecting the existing `extract_crops()`
function. Then fix thresholding, line segmentation, component grouping,
and duplicate-box handling in that order. Only move to a learned
character detector if the evaluation shows that touching or stylized
characters cannot be separated reliably with classical methods.
