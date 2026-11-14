# JPEG compression history and selected source blocks

The existing aligned JPEG check can now show a source crop and overlay of
the connected blocks that passed its local rule. Each marked block covers
8 × 8 decoded pixels. The overlay follows the original picture's EXIF
orientation. The evidence card records the block count, native coordinates
and source identity. The PDF and JSON refer to the same evidence. When the
check selects no region, the generic heatmap stage no longer substitutes
the highest response values. If partial edge blocks prevent an exact grid
mapping, the check retains its measurement and omits the precise outline.
Ordinary blur, copying, re-export and generated edits can all produce this
compression-history result. Its cause remains unresolved.

## Controlled JPEG comparison

A new experiment estimates whether the coefficient values in two
directions fit an earlier standard JPEG quantization table. It accounts
for the residues that the delivered table permits. Nonzero coefficients
from complete native cells support a prior quality only when both
directions support the same pattern across the image. The proposed
region is a compact rectangle with weaker support than its surrounding
ring in both directions. The experiment uses single JPEG uploads and
no model weights. It estimates compression history and carries zero AI
confidence weight. The method draws on work about local JPEG histories
and compatibility. [JPEG ghost study](https://hfarid.org/downloads/publications/tifs09.pdf)
· [Dual JPEG compatibility study](https://arxiv.org/abs/2408.17106)

The 26 credited photos come from the pinned BG-REAL panel used in the
earlier PNG study. The edit families are LaMa removal, Stable Diffusion 1.5
insertion and Kandinsky background replacement. Eight photographers supply
ordinary calibration views. Eighteen other photographers supply the held
views. Each held photo has sharp and feathered edits, unchanged, local
blur, tone and copy, a global resize round trip and a prior JPEG save.
Each final file uses 4:2:0 JPEG at quality 75 or 90. In a separate stage,
every source is first saved at a deterministic quality of 50, 55 or 60.
The same local generated region is then composed over that decoded copy.
The reference mask comes from the exact changed pixels before final
encoding. Both workflows share underlying photos. The publisher photos'
earlier encoding history is unknown. [Source qualification, credits and license](image-edge-context-new-sources-2026-11-14.md#photo-qualification)

In each delivery profile, the candidate score must strictly exceed the
second-highest calibration source maximum or 0.24, whichever is greater.
Useful localization requires 30% intersection over union and half the
changed pixels. Advancement requires useful gains on at least three new
sources across two edit families, with at most two additionally flagged
ordinary source groups and four misplaced edits. The comparison keeps
an existing aligned JPEG selection when one exists. These results apply
to the aligned JPEG check. Other image detectors were outside the panel.

| Held delivery | Existing useful locations | Existing ordinary highlights | Added useful locations |
| --- | ---: | ---: | ---: |
| Direct JPEG75 | 0/36 | 4/108 | 0 |
| Edited after JPEG, final JPEG75 | 34/36 | 57/108 | 0 |
| Direct JPEG90 | 0/36 | 2/108 | 0 |
| Edited after JPEG, final JPEG90 | 34/36 | 58/108 | 0 |

The experimental check found a widespread candidate pattern in only four
of 576 held files, all staged JPEG90 deliveries. None matched the authored
earlier quality. Its four proposed rectangles supplied zero useful locations
before the final cutoff. The experiment stays offline. The existing check
highlighted all 18 local blurs and all 18 local copies per staged profile.
These known ordinary edits explain why the deployed evidence card describes
compression history and leaves the origin of the difference unresolved.
[Input commitments and per-file counts](evaluation-panels/image-jpeg-history-gap-2026-11-14.json)

## Local app verification

Two isolated simulated USD 9.99 purchases exercised the Full image
worker with the aligned JPEG check enabled. Neither analysis started
before confirmed payment. The edited image used EXIF orientation 6 and
selected 383 source blocks at x 264, y 296, width 136 and height 184.
The Evidence API and JSON agreed with the signed PDF. The decoded
original crop matched the submitted image pixel for pixel, and the
ordinary re-export produced no region. The report regression also
exercises a partial-block file and verifies the exported binary mask.
[Local checkout receipt](evaluation-panels/image-jpeg-history-app-2026-11-14.json)

## Replay

The standard-library reader recomputes the points, localization, ordinary
flags, source separation and advancement decisions. The native replay
compares 768 delivery and truth hashes, existing outcomes and proposed
rectangles. Two local runs using separately acquired copies of the pinned
publisher files reproduced every measurement. The GitHub receipt includes
scalar counts and hashes. Source media come from the photographer-attributed
publisher paths in the parent receipt.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/image-jpeg-history-gap-2026-11-14.json
parent=docs/evaluation-panels/image-edge-context-new-sources-2026-11-14.json
python3 -B -S -m scripts.image_jpeg_history_gap_panel review "$receipt"
backend/.venv/bin/python -B -m scripts.image_edge_context_new_sources hydrate "$photos" --receipt "$parent"
for phase in freeze measure replay; do
  backend/.venv/bin/python -B -m scripts.image_jpeg_history_gap_panel "$phase" "$run" --sources "$photos" || exit 1
done
cmp "$run/receipt.json" "$receipt"
```

Run the report regression from `backend` with
`python -m pytest -q tests/test_image_evidence_reports.py -k jpeg_history`.
