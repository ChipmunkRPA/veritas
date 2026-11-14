# JPEG90 earlier-copy confirmation on 34 additional photographs

The previous test found two additional edit regions on one source photo.
This confirmation fixes the same source requirements, thresholds and
measurements and compares both proposed routes with the full earlier-copy
check. One route runs after an unassessed comparison. The second also
runs when an assessed pair has no selected region. Both leave existing
highlights intact. [Previous comparison](image-reference-same-jpeg-2026-11-13.md)

## Sources and controls

Use every qualified source with both dimensions from 512 to 768 pixels
from three retained BG-REAL panels. A source shared by the external SDXL
and Kandinsky cases contributes only its earlier planned SDXL edit. The
source roster contains 13 LaMa, 14 Stable Diffusion 1.5, six Kandinsky and
one SDXL photograph. The selection excludes the 37 previous exact-block
comparison photos by photo ID, photographer ID, credited photographer
name and original file digest. The new bank has no duplicate decoded
original or near-identical informative thumbnail within the selected
set. The earlier studies used these photographs to investigate other
checks, including JPEG texture and standalone region detection. No
inference, model weights or fresh photo downloads are required.
[First retained source bank](image-three-class-patch-fresh-2026-11-12.md)
· [BG-REAL attribution and rights](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

Each photograph supplies a sharp and feathered generated edit, three
ordinary local edits, five unchanged or global-processing controls,
and four incompatible comparisons. The local controls use the same
pixel support as the known generated edit. The nuisance controls cover
an earlier JPEG save, resizing, global tone and blur. The incompatible
controls cover a different photo, a fractional or integer shift and a
reflection. All inputs are encoded once at baseline JPEG90 4:2:0 after
the planned operations. Truth is the exact pre-encoding changed pixels.

A useful region needs at least 50% edited-pixel coverage and 0.30
intersection over union. Advancement requires at least two useful
additions from two separate photographs, zero additional highlights on
untouched or global-processing controls, zero new assessed incompatible
pairs and zero additional misplaced edit regions. Each sharp and
feathered variant counts separately for deliveries and once for source
breadth. Both routes supply zero AI-confidence weight.

## Results

| Check | Useful generated locations | Useful ordinary local locations | Unchanged or global-processing highlights | Incompatible pairs assessed |
| --- | ---: | ---: | ---: | ---: |
| Complete current comparison | 64/68 | 59/102 | 32/170 | 0/136 |
| Exact fallback after unassessed | 65/68 | 61/102 | 32/170 | 0/136 |
| Exact fallback also after empty result | 65/68 | 66/102 | 32/170 | 0/136 |

The fallback added one useful feathered edit from LaMa. The sharp
variant of that photograph had a new mask too. Of its 4,441 selected
pixels, 4,430 fell inside the known edit, giving 0.489 intersection
over union and 48.95% coverage. The frozen 50% coverage requirement
counts that mask as a non-useful addition even though it points to
the correct area. The remaining two incumbent misses come from one
Kandinsky photograph. They highlight parts of its edit and cover
only 20.82% and 23.97% of the known changed pixels.

The proposed routes retain all incumbent highlights and add no flags
on the 170 untouched or global-processing pairs. All 34 untouched
files had empty masks. Neither route meets the frozen source-breadth
and no-misplaced-additions requirements. The broader rule located
seven more ordinary local changes and remains experimental.
[Confirmation ledger](evaluation-panels/image-reference-same-jpeg-confirmation-2026-11-13.json)

## Local option

Across the two source-separated held panels, the narrower fallback
improved useful localization from **108/114 to 111/114** across two
newly located source photos. It added no highlights across **285**
untouched or global-processing deliveries and no new assessments
across **228** incompatible deliveries. The fourth added region was
the sharp LaMa edit above, with **99.75%** of selected pixels inside
the edit. The combined totals support a development opt-in for
recovering physical differences in otherwise unassessed pairs. Each
separate panel retains its original failed release decision. These
tests do not measure general JPEG or AI-origin accuracy.

On the local API, enable `VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED`
and `VERITAS_IMAGE_REFERENCE_SAME_JPEG`. From a completed image
analysis, choose **Compare with an earlier copy** and upload the earlier
JPEG. The route applies only if earlier comparisons could not assess
the pair, the dimensions match and each side is 512–768 pixels, and
both files are upright, unprofiled RGB JPEG90 with matching 4:2:0 sampling
and standard quantization. Supported encoding now includes
[single-frame progressive JPEG](image-reference-progressive-2026-11-14.md).
Matching detailed blocks
support the common framing and selected differences are drawn on the
delivered pixels. The browser, PDF and JSON share exact native crops,
an overlay and the alignment counts. The finding contributes zero AI
confidence. The option defaults to `false` and applies only to local
development. Set it back to `false` to restore the earlier routing.

An isolated worker completed the newly found feathered edit after
simulated $9.99 payment. The unpaid and repeated-payment events
created no new analysis or job. The two source hashes and six exact
crops replayed against the original files. The selected mask retained
3,913 pixels, including 3,894 known changed pixels. Chrome showed
seven images and the toggleable overlay, downloaded a matching JSON
and PDF, and raised no page error. Disabling the option reproduced
the unassessed response and preserved the completed analysis.
[Local app verification](evaluation-panels/image-reference-same-jpeg-app-2026-11-13.json)

## Replay

Use the retained source directories. The native runner checks each
private roster against the existing public intake commitments. It keeps
rendered JPEGs and overlays in a temporary directory and records only
input hashes, source commitments, and overlap counts. The integrated
route reproduced every one of the 476 pre-integration rows. A second
native run reproduced the full receipt. Standard-library results
agree on Python 3.11 and 3.12 and reject impossible block counts.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze score replay; do
  backend/.venv/bin/python -B -m scripts.image_reference_same_jpeg_confirmation "$phase" "$run" \
    --primary "$primary_roster" --backup "$backup_roster" --wavelet "$wavelet_plan" \
    --flux "$flux_inputs" --patch "$patch_bank" --texture "$texture_bank" \
    --editors "$editor_bank" || exit 1
done
python3 -B -S -m scripts.image_reference_same_jpeg_confirmation review \
  docs/evaluation-panels/image-reference-same-jpeg-confirmation-2026-11-13.json
```
