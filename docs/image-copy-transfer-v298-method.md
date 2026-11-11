# Copy corroboration on eight held-out licensed photos

This model-free experiment transfers the fixed v295 same-image donor cue to
eight retained v39 CC BY 2.0 source groups. The historical attribution and
exposure receipts leave 81 eligible groups and 83 eligible files after the
v276 and v285 photo groups are excluded. The corresponding v280 and v288
detector comparisons consumed only those excluded groups. Use one file per
source group and eight distinct recorded Flickr profiles. These groupings
still have the original limits for unknown aliases, transformed duplicates,
unrecorded exposures, and foundation-model pretraining.

## Freeze before opening selected source files

Use the fixed v298 salted encoded-digest order within and across components.
Freeze the first eight groups with their attribution, source digests,
geometry, and software versions before this study opens source bytes.
Assign two groups to each combination of interior or left edge placement and
aligned or shifted JPEG grid. The source crop is the prior v276 unscaled
256×256 center crop. The placement and donor coordinates are in crop pixels.
Every 64×64 donor is disjoint from its recipient. Donor coordinates
`(176, 176)` and `(171, 179)` test aligned and shifted JPEG-grid phases
respectively. Each group gets all variants in both formats.

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_v298() { backend/.venv/bin/python -B -m scripts.image_copy_transfer_v298 "$@"; }
run_v298 preflight
run_v298 freeze
# Replace the value with the exact digest printed by the freeze command.
freeze_sha="printed-digest"
run_v298 run --freeze-sha256 "$freeze_sha"
```

Preflight and freeze read source metadata and check file paths and sizes.
Freeze writes only to a new ignored evaluation directory. The run verifies
the freeze digest, the exact regenerated plan, the 5 MB encoded-source bound,
and the native upright JPEG dimensions and RGB digest before it takes the
planned crop. It processes one source at a time with one OpenCV thread.
The readout keeps all 80 planned views in the denominator. Input errors keep
their source groups in the aggregate and mark the run incomplete. There are
no replacements.

## Paired controls and readout

Each source has an original, contrast and color change, blur, translated copy,
and horizontally reflected copy. The two copy variants share the same known
donor. A fixed 64×64 recipient mask represents an already localized candidate
in every view, including the original. Each variant's exact changed RGB mask
is derived separately against the crop before encoding. The lossless PNG and
whole-image JPEG90 views share that native truth. Compare JPEG variants
against the identically encoded original and count extra differences outside
native truth. Keep the proposal mask unchanged even if the ordinary edit has
unchanged pixels inside it.

Report exact donor location and transform recovery for the two copy
families, other supported donors, ambiguity, no donor, and unassessed
geometry or texture. Keep native and JPEG90 outcomes, placements and grid
phases identifiable. Retain the source attributions, input fingerprints
and per-view outcomes in the private trial. The public readout may use the
aggregate counts and hashes.

All detector proposal outcomes are explicitly unassessed. This round tests
copy corroboration at a known region and includes no AI-generated edits.
Fresh generated or mixed edits and actual detector proposals require a
separate comparison. This cue does not change evidence highlights,
authorship or confidence, or the serving configuration.
