# Whole-image PNG screening with a layout scope check

Chipmunk Verify's optional Full image review now supports a single photographic
PNG upload. A native-pixel scope check skips common interface and document layouts.
For a positive response, the browser and reports show the exact three inspected
views and their source positions. The supplementary response adds zero weight to
overall confidence. A response can leave an otherwise original-leaning result
inconclusive when the two checks disagree.

The change addresses a measured failure in the earlier PNG trial. The existing
whole-image check flagged 6 of 8 ordinary browser captures. An existing global
flatness rule excluded those simple layouts but only two of 12 later captures
with large photos. The new method checks local raster geometry, so a full-bleed
photo can have an interface header or small overlay and still be out of scope.
[Earlier native PNG and layout tests](image-whole-png-transfer-2026-11-12.md)

## Frozen comparison

Reuse the existing whole-image model's three-view operating point. Measure the
share of native 8 × 8 blocks with exactly one RGB value, long horizontal or
vertical runs of exact RGB changes, and common exact ink colors across recognized
words. A long edge covers at least 80 pixels and 10% of the corresponding side.
A word supports a common ink color when it has sufficient contrast against its
surrounding pixels and that color covers at least 4% of the word box. A line
with the same ink color in three or more recognized words supports a skip.
The strict existing flatness limit, a qualifying edge, or common ink is sufficient.
The lower global flatness rule is a separate comparison. Text strings are
used inside the OCR process and its local geometry filter. The study and
app keep only the scope decision or numeric measurements.

This approach is informed by work showing that ordinary text-rich renders and
image-model output are a different detection domain from photographic scenes.
The TextFake paper's dataset is noncommercial and was not acquired. The
[second text-rich image study](https://arxiv.org/html/2606.19259v1) reports
substantial category dependence on posters, tables and screenshots. The
[November inpainting study](https://arxiv.org/html/2602.00192v2) also shows
why a whole-image signal cannot establish which pixels were generated. Its
code and data were not used. [TextFake paper](https://arxiv.org/html/2606.01050v1) ·
[TextFake dataset terms](https://huggingface.co/datasets/Yuning0123/TextFake)

The 110 files consist of 36 development PNGs from the earlier test, 24 retained
photo and browser views, 24 new photo and browser views, eight newly acquired
native PNGs, and 18 lossless PNG encodings of earlier generated JPEGs. The 18
JPEG equivalents preserve decoded upright RGB exactly and retain earlier source
exposure. The eight new native PNGs are separate publisher scene IDs from the
CC BY 4.0 CompoSET release. Exact encoded and RGB checks plus a difference
hash exclude measured duplicates in generated cohorts. The fresh capture bank
uses six new authored layouts with two native Chrome resolutions per layout.
The photographic controls use a separate set of attributed Wikimedia photos
from the existing licensed archive. None of their publisher hashes recurs in
the earlier study's photographic controls. They have earlier project exposure.
[Exact source receipts](evaluation-panels/image-png-layout-guard-2026-11-12.json) ·
[CompoSET release](https://huggingface.co/datasets/CompoSET/CompoSET/tree/d428bf4f23f50ababb205185f179737cd9638db2) ·
[CompoSET generation method](https://arxiv.org/html/2610.41118) ·
[Licensed JPEG source roster](evaluation-panels/image-whole-fresh-sources-2026-11-12.json) ·
[Detector and base terms](image-smaug-v7-preparation.md) ·
[Pinned OCR notice](https://github.com/tesseract-ocr/tessdata_fast/blob/87416418657359cb625c412a48b6e1d6d41c29bd/LICENSE)

## Held layout and detection results

| Cohort | Files | Unguarded elevated responses | Elevated responses with flatness floor | Elevated responses with layout check |
| --- | ---: | ---: | ---: | ---: |
| Earlier JPEGs encoded as PNG | 18 | 18 | 18 | 18 |
| New native generated PNGs | 8 | 4 | 4 | 4 |
| Ordinary photo views, new and retained | 24 | 0 | 0 | 0 |
| Retained browser layouts | 12 | 4 | 2 | 0 |
| New browser layouts | 12 | 3 | 1 | 0 |

The layout check excluded all 24 held browser views and none of the 24 matched
ordinary photo views. All 26 generated files remained in scope. Four of eight
new native generated images remained below the same response point after scope
admission. This supplies a new delivery route for photograph-like inputs with
limited evidence on native newer generators. The browser panels use authored
layouts with known ordinary sources. A text-rich image can be synthetic and
still fall outside scope. Unknown interface styles, layout language, raster
scaling and in-the-wild errors remain unmeasured.

The isolated CPU Full baseline completed all 74 held files on the same PNG
bytes. It already gave an AI direction on all eight new native PNGs. On the
18 earlier JPEG equivalents, Full gave six AI directions and the supplement
added 12 indications across three generators. Two of the 12 disagreed with
an original-leaning Full result. The supplement introduced zero ordinary
flags. One retained photographic view already received an AI direction from
the existing Full check. The additional value shown here is corroboration
on native inputs and better coverage of known generated images delivered as
PNG. The first baseline bootstrap stopped before inference on a snapshot path
error, and a second attempt stopped during MPS model activation. The CPU run
used the same source inputs and committed results for all 74 files. Its public
lock omits machine-specific paths while retaining the private lock hash.

## Local app and native verification

Enable `VERITAS_IMAGE_WHOLE_PNG_REVIEW_ENABLED=true` alongside the configured
whole-image runtime in a development instance. PNG must be upright, single
frame, opaque 8-bit true color, at least 768 pixels on the short side, and
at most 4,920 pixels on the long side. The JPEG rules keep their existing
bounds. The OCR step uses the existing local pinned English language data.
It has a bounded process, output and word work budget. If OCR is required and
unavailable, the supplementary check is skipped. The explicit layout skip
appears in the same evidence section as a result. It carries no authenticity
judgment.

Three simulated prepaid Full uploads reproduced a positive native PNG, an
ordinary photographic no-flag, and an ordinary table-layout skip. Checkout and
unpaid events created zero analysis jobs. In the positive case, the three
528 × 528 views had the same pixel hashes in the evidence API and Standard
PDF. All three reports agreed with the JSON export and verified as signed
evidence packages against the submitted file. The supplementary crops are
embedded in the signed PDF. The signed package's separate image-region
assets continue to follow their existing source bindings. Chrome loaded
all three results, checked the three crop dimensions and the explicit
scope reason, and recorded zero page errors.

A separate native readback verified all 110 delivered PNG and RGB commitments,
110 exact app scope decisions, 220 longest-edge measurements, and 222 exact
inspected crop pixels and source geometries. A second Chrome capture reproduced
all 12 new browser PNGs byte for byte. The public replay checks pinned sources,
raw response arithmetic, scope decisions and reported group counts without
image data or detector weights. There is no third-party reviewer attestation.

## Reproduce

Run the capture plan against the retained credited photo archive. The photo
source and crop coordinates are recorded in the freeze. Fetch the eight
exact CompoSET files from the recorded links and verify each receipt hash.
The existing native PNG and retained browser probe documentation describe
how to reconstruct those prior inputs. Use the pinned whole-image runtime
and provisioned English language asset.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_png_layout_guard_transfer capture-plan "$run"
node backend/scripts/capture_image_png_layout_controls.cjs "$run"
backend/.venv/bin/python -B -m scripts.image_png_layout_guard_transfer freeze "$run" \
  --publisher "$publisher" --scope "$scope"
backend/.venv/bin/python -B -m scripts.image_png_layout_guard_transfer measure "$run" \
  --tessdata /opt/homebrew/share/tessdata
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 tmp/image-smaug-v7-env/bin/python -B \
  -m scripts.image_png_layout_guard_transfer candidate "$run"
backend/.venv/bin/python -B -m scripts.image_png_layout_guard_transfer combine "$run"
python3 -B -I -S backend/scripts/review_image_png_layout_guard.py \
  docs/evaluation-panels/image-png-layout-guard-2026-11-12.json
```

To repeat the paired Full baseline from the same input freeze, use the exact
tracked snapshot below, the existing commercially scoped model directory, and
an initialized local `backend/data/veritas.db` with the recorded policy row.
The baseline reads that policy and runs each case in an isolated copy. Use the same
local image-related settings that match the receipt's model and configuration
commitments. Keep the CPU run's isolated directory and pinned checkout together.

```sh
mkdir "$run/checkout" "$run/mpl"
git archive c8ff179fd9a3e9289fe06b772e1c4e351bf6778c \
  backend/app backend/alembic backend/alembic.ini | tar -x -C "$run/checkout"
git rev-parse c8ff179fd9a3e9289fe06b772e1c4e351bf6778c > "$run/checkout/commit.txt"
PYTHONPATH="$run/checkout/backend:$PWD/backend" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  VERITAS_BILLING_MODE=disabled VERITAS_DEVICE=cpu \
  VERITAS_IMAGE_WHOLE_REVIEW_PYTHON="$run/absent-whole-image-worker" MPLCONFIGDIR="$run/mpl" \
  backend/.venv/bin/python -B -m scripts.image_png_layout_full_replay full "$run"
backend/.venv/bin/python -B -m scripts.image_png_layout_full_replay export "$run"
python3 -B -I -S backend/scripts/review_image_png_layout_guard.py "$run/full-receipt.json"
```
