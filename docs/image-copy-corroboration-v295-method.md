# Same-image copy corroboration on authored regions

This standalone experiment looks for another location in the submitted image
that matches an existing proposed region. An exact donor can help explain an
ordinary clone or retouch. The paired regions establish a pixel similarity.
The direction of the edit, authorship, and the authenticity of other parts of
the image remain unknown. The existing highlight and its exact mask are retained
in every outcome.

## Authored qualification

Use a single CPU thread and in-memory authored 256×240 RGB images. For every
proposal use native pixels and a JPEG90 encoding of the entire image. The
translations and reflections share known donor and recipient rectangles.
The mixed fixture has a reflected copy in one region and an independently
authored procedural replacement in another. Both proposals are tested on the
same submitted image. The procedural patch checks separation of the two
proposals and supplies no AI performance data. Periodic repetition, a flat
region, and an untouched textured region with a candidate highlight are
independent controls.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_copy_corroboration_v295 authored
backend/.venv/bin/python -B -m scripts.image_copy_corroboration_v295 protocol
backend/.venv/bin/python -B -m pytest -o addopts= -q -p no:cacheprovider backend/tests/test_image_copy_corroboration_v295.py
```

The authored result records the two native-coordinate boxes when a donor is
unique, the transform, exact proposal digest, anchor support, coverage and
overlap with each authored footprint. The protocol command prints the code
digests, constants, and dependency versions for review.

The local check recovered the known donor in all six native or JPEG copy
views, retained all 14 proposed masks, and called both repeated-pattern views
ambiguous. It found no donor in either encoding of the independent replacement
or the untouched textured region. The two flat views remained unassessed.
Two executions produced identical readouts. [Authored aggregate](evaluation-panels/image-copy-corroboration-v295-authored.json)

## Matching rule for a prospective panel

Work in upright source coordinates up to 512 pixels per side. Reuse the
repository copy-move luminance conversion at native size. Sample a fixed
3×3 or sparser grid with 16×16 anchors at least four pixels inside the
proposal box. Require each anchor to lie completely in the exact proposal
mask and have luminance standard deviation of at least eight. A proposal
with fewer than four eligible anchors is unassessed.

Evaluate a common integer translation and horizontal reflection separately.
Search all fully in-frame donor-box origins outside the proposed region's
bounding box. At least two-thirds of anchors must have normalized correlation
at or above 0.96 at their expected positions. Rank those qualifying origins by
median correlation over all anchors. A second donor with box IoU below 0.50
and a correlation gap under 0.03 yields an ambiguous result. With no eligible
donor, report no donor found. With a unique donor, record its box and transform
as copy-compatible evidence. Correlation measures pixel similarity. It carries
no authorship probability.

Published copy-move evaluations motivate checking transforms and compression
separately. Research on handcrafted and synthesized copies also motivates
treating the donor as pixel evidence in either case. [Copy-move evaluation](https://arxiv.org/abs/1208.3665) · [Handcrafted and synthesized copies](https://arxiv.org/abs/2512.07110)

Use a new source-group roster for evaluation, with original images and all
derived variants kept together. Audit the retained v39 commercial-use records
and subsequent exposure inventories through v288. Exclude v276/v285 sources
and all other known detector, calibration, or cue-development source exposures.
Before looking at scores, record the exact source digests, crop geometry,
rights, variants, checkpoints used to produce any proposals, this
implementation digest, and dependency versions in the prospective evaluation
manifest. Report any unresolved lineage gaps.

Use the same target geometry for the licensed generated composite and its
paired unchanged, tone, blur, translated-copy, and reflected-copy variants.
Include a distinct nonoverlapping ordinary-copy plus generated-edit view,
and keep their exact changed RGB masks separate. Apply native PNG and
whole-image JPEG90 uniformly across variants. Retain native pre-encoding truth
for both formats and record JPEG-only pixel changes separately. Keep source
media and per-source maps private.

Run the fixed method on exact control masks as a localization-conditioned
test. Evaluate saved baseline detector proposals separately to measure
whether the method explains actual highlights. For the detector run, score
every existing proposal and record groups with no proposal explicitly.
Compute overlap with ordinary-only and exact generated footprints after
matching. Count ordinary exact-donor recovery, proposals incorrectly attributed
to a donor on generated regions, overlaps in mixed images, ambiguities on
repeated textures, donor-free proposals, and unavailable regions separately.
Summarize native and JPEG90 by source group and variant, including expected and
completed coverage. Keep settings fixed through that comparison.

The authored qualification has no serving integration. Every detector
proposal remains visible regardless of the donor outcome, and this cue
contributes no authenticity verdict or change to the overall score.
