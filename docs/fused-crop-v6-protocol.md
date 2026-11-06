# FUSED crop v6: consumed-source development protocol

Prepared 6 November 2026, before this crop method runs on project images.
This is a separate hypothesis from v5's recompression-feature tiling. It uses
the existing licensed inpainting checkpoint and semantic backbone, with their
unchanged independent float32 runtime. No new weights or training are involved.
Neither the detector registry nor customer configuration enables this path.

The currently selected format-specific components remain the controls: the
PNG recompression check and the full-image FUSED check on JPEG. A crop may supply
regions only when the assessed control displays none. A missing or failed
control stays unassessed. Existing highlights are preserved exactly.

## Inputs and geometry

Development reuses exactly the 24 consumed IML development source pairs: eight
each from Firefly, SD2 and FLUX Fill. The original 72-pair TGIF manifest is pinned
as `4ec157d240f88fce43e554d5dee0bd2da9ac388345d3f0a188c93826629c8230`;
the old 24-group selection receipt is pinned as
`36a7c32af569d912b0d3737c2486092d4cf37469c2a652e8bd5acfa1cf55e04d`.
There are 48 source images, each scored natively and after whole-image JPEG75:
96 planned input outcomes and four scheduled crops per eligible input. These
are 24 independent source groups, not 96 independent examples.

Each source at least 320 pixels on both axes supplies four corner crops, each
60% of source width/height rounded upward. Crops are taken from already-oriented
native RGB, after any declared whole-image JPEG variant. They are not separately
JPEG encoded. The unchanged predictor resizes each crop to 512 square, as it
does for its full-image input. Each 512-square native response and uncalibrated
image score is retained, with its exact source box and cropped RGB SHA-256.

Maps are bilinearly projected into source coordinates. Six percent of each
crop's internal border is suppressed; image borders are retained. Eligible maps
are combined by pixelwise maximum. Source reliability exclusions, screening
grid quantization, threshold 0.5, component filtering and the existing two-region
per-map cap are applied using the full source image. Cropping changes model
context and can create false highlights; it is a hypothesis to test.

## Development rule and next decision

For native and JPEG75 separately, select the smallest crop image-score gate at
or above 0.5 that adds regions on at most one of the 24 genuine controls. Evaluate
every score transition after actual display filtering; do not assume component
filtering is monotonic. Edited labels never select that gate. All maps and
measurements remain unchanged. The reviewer must reconstruct input bytes, crop
RGB commitments, native arrays and baseline displayed metrics before selection.
Incomplete extraction cannot fit a gate. Failures remain recorded in the full
planned denominator; do not replace or silently discard inputs.

Development must recover more overlapping edits and higher mean edited IoU
with at most one additional genuine highlight per format before a format may
advance to separately frozen, source-disjoint confirmation. No confirmation
entrypoint is provided here. Do not inspect unused images to rescue a failed
development result. Report per-generator outcomes, misses, false highlights,
unassessed cases and source-paired uncertainty. Misses contribute zero IoU.

The primary source-pair RGB-difference masks indicate pixels changed between
the two supplied images. They do not independently establish semantic editing
boundaries or prove AI authorship. Preserve this distinction and available
publisher masks in future confirmation. Unknown pretrained-source overlap
remains a limitation. No crop score is a calibrated probability, and no result
enters overall confidence or permits service adoption on development alone.

## Commercial assets

Use the existing MIT FUSED release at `aonuzhdin/FUSED`, revision
`b7dfe0cb2d2f8fc8fd9e6ebea5f7185cfa8e1af7`, with checkpoint SHA-256
`7e527d49080a773641cf96915b4b3fa12d571c5b87f35c527f13ab82f148364c`.
The existing Apache-2.0 ConvNeXt semantic backbone SHA-256 is
`1ddf6dbc7f0753fc4dfec5db5fa474acfd1ac6f67f466d6a976780c30b3a1b23`.
Retain existing upstream notices and source terms. No raw benchmark media,
customer file, model weight or private path is published by this experiment.
