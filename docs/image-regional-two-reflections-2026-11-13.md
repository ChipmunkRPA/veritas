# Reflected views for small image edits

Chipmunk Verify's image evidence now shows the source coordinates and
overlap counts for each candidate reviewed by the horizontal repeat,
including candidates the repeat withheld. The browser and Standard PDF
show the aligned binary repeat mask already stored with the evidence.
White cells mark the repeated response on the submitted image grid.
The separate reliability mask marks where the source could be measured.

Two isolated prepaid Full analyses reran a known edit and its untouched
match through the real regional detector. The edited image kept one of
two original candidates. The untouched image kept neither. For both,
the coordinates and overlap counts matched the API and JSON, the PDF
embedded the stored mask pixels, and the signed package verified both
overlaps against the submitted image. This matched pair tested the
application workflow. [Earlier PNG evaluation and source terms](image-regional-flip-confirmation-2026-11-13.md)

## Comparison

The local experiment reopened the same 108 source-bound PNG deliveries
from 28 photos. They include 40 sharp or feathered edits from four
editor families and 68 untouched or ordinarily processed controls.
The runner uses the existing commercially permitted regional checkpoint
without fitting or changing its point. Prior exposure means this
comparison tests a specific change against an existing benchmark.

The first arm keeps an original component when either horizontal or
vertical reflection supports at least half its measured cells. The
second leaves the horizontal baseline in place and seeks a supplemental
region only when the baseline selected nothing. It requires a component
from each reflected view to share at least half of both reliable areas.
It returns their intersection in submitted-image coordinates. No reflected
view is an independent model. The method examines localization
repeatability, without estimating AI authorship. Recent research on
inpainting motivates testing localization against ordinary edits even
when a detector identifies the broad image distribution.
[Inpainting Exchange study](https://arxiv.org/abs/2602.00192v2)

A useful location covers at least half the known changed pixels and
reaches 0.30 intersection over union. The frozen comparison requires at
least three added useful locations across two editor families, no lost
useful locations, at most two additional ordinary source groups and at
most two new misplaced edited-file highlights.

## Results

| Method | Useful edit locations | Ordinary highlights | Misplaced highlights on edited files |
| --- | ---: | ---: | ---: |
| Existing horizontal repeat | 12/40 | 26/68 | 9/40 |
| Keep original regions supported by either repeat | 11/40 | 31/68 | 12/40 |
| Add a region where both reflected masks agree | 12/40 | 27/68 | 11/40 |

The either-repeat arm found no new useful locations and lost one by
retaining a second off-target region. The two-reflection proposal
covered 1,504 of 9,033 changed pixels on one missed edit, below the
localization requirement. Its other added edited-file highlight covered
zero changed pixels. Both arms stay experimental. [Pixel-count ledger](evaluation-panels/image-regional-two-reflections-2026-11-13.json)

## Replay

The standard-library reader independently recounts the public ledger
against the earlier frozen source and horizontal-mask records. Native
inference can reconstruct the same source PNGs and exact changed-pixel
masks from the earlier replay inputs.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -B -S -m scripts.image_regional_two_reflections review \
  docs/evaluation-panels/image-regional-two-reflections-2026-11-13.json
backend/.venv/bin/python -B -m scripts.image_regional_two_reflections freeze "$trial" --bg "$insertions" "$remote_context"
backend/.venv/bin/python -B -m scripts.image_regional_two_reflections measure "$trial" --bg "$insertions" "$remote_context"
```
