# Locating earlier JPEG grids after recompression

The shifted-grid check located eight of 48 edits delivered at JPEG 90. All eight locations recovered the known earlier grid phase. The aligned check located 46 of 48 edits, including all eight found by the new method. The shifted-grid check flagged fewer controls at JPEG 90 and located none of the same edits at JPEG 75. Neither delivery met the frozen advancement rule. [Aggregate and source-group counts](evaluation-panels/image-shifted-jpeg-likelihood-2026-11-11.json)

## What was measured

The study used the luma pixels and quantization table of each delivered JPEG. It searched all 63 nonzero 8 × 8 grid offsets and five standard candidate earlier tables. At each candidate grid, a clean-room implementation projected quantization error from the overlapping final JPEG blocks onto eight low-frequency DCT coefficients. It used the projected error to compare a wrapped-normal density around earlier quantization multiples with a uniform phase. The check excludes near-zero coefficients and steps too fine relative to projected error. This measures periodic earlier quantization, distinct from the aligned check and the prior PNG cue that counts near-zero coefficients and corroborates block-edge steps. [Bianchi and Piva’s non-aligned compression model](https://iris.polito.it/retrieve/e384c42e-2465-d4b2-e053-9f05fe0a1d67/bian_TIFS2012_OA.pdf) · [Previous grid comparison](image-local-grid-transfer-2026-11-11.md)

Each candidate location had to show positive local likelihood and stronger evidence than its surrounding ring. The reported mask covers the winning rectangle in the JPEG’s stored pixel coordinates. For each delivery, the strict review point was the second-highest of eight source maxima from 40 development controls. The source panels, likelihood, mask geometry, point rule and advancement rule were fixed before this run read the confirmation pixels. [Frozen profile](evaluation-panels/image-shifted-jpeg-likelihood-freeze-2026-11-11.json)

The two confirmation panels contained 12 source groups each, with disjoint attributed photos and creators from each other and the eight development groups. The runner checked the exact earlier intake receipts, regenerated each JPEG and exact precompression truth mask, and verified them against both prior sealed comparison journals. Each format had 24 sharp and 24 feathered inserts with an intentionally shifted first grid, plus 120 controls. The controls covered originals, a whole-image grid shift, ordinary contrast and blur, and an aligned patch. A useful mask had at least 0.30 intersection over union and 0.50 recall. BG-REAL records CC BY 2.0 for these selected source photos and CC BY 4.0 for publisher contributions. Source media, identities and row-level locations stay in the local archive. [Development sources](image-bg-real-insertions-2026-11-11.md) · [Earlier confirmation sources](image-background-patch-transfer-2026-11-11.md) · [Later confirmation sources](image-wavelet-region-transfer-2026-11-11.md) · [Prior JPEG recompression comparison](image-multiscale-jpeg-transfer-2026-11-11.md)

## Results

| Delivery | Check | Sharp useful /24 | Feathered useful /24 | Control flags /120 | Control groups flagged /24 |
| --- | --- | ---: | ---: | ---: | ---: |
| JPEG 90 | Shifted-grid likelihood | 7 | 1 | 11 | 3 |
| JPEG 90 | Existing aligned check | 23 | 23 | 60 | 23 |
| JPEG 90 | Aligned plus recompression corroboration | 16 | 14 | 19 | 10 |
| JPEG 75 | Shifted-grid likelihood | 0 | 0 | 23 | 11 |
| JPEG 75 | Existing aligned check | 4 | 4 | 27 | 12 |
| JPEG 75 | Aligned plus recompression corroboration | 0 | 0 | 6 | 5 |

The JPEG 90 shifted-grid check flagged two unchanged originals and two whole-image shift controls. It produced six useful locations on the earlier panel and two on the later panel. The results cover constructed shifts at a fixed pair of prior qualities, with repeated variants within each source group. They measure local compression history. Ordinary edits and saving operations can create the same signal. The pilot remains experimental with zero weight in the app's AI confidence calculation.

## Local reproduction

Run the three stages from the repository root using the retained development and confirmation intakes, the previous local-grid parity and multiscale evaluation directories, and a fresh output directory. The runner writes receipts and local coordinates to the fresh directory. The confirmation processed 336 views on CPU in 111.9 seconds. The six output files occupied less than 1 MB. A separate Pillow replay checked all 336 input digests and candidate mask digests, recomputed 1,008 overlap measurements, and matched all 12 panel, delivery and method summaries.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze fit confirm; do
  backend/.venv/bin/python -B -m scripts.image_shifted_jpeg_likelihood "$stage" \
    "$development" "$earlier" "$later" "$prior_grid_parity" "$prior_recompression" "$output" || break
done
```
