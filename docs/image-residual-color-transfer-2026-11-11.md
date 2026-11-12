# Residual color comparisons on small image edits

Three local color-response checks found zero useful sharp or feathered edit regions on two separate 12-photo panels. The two-band check flagged 32 of 144 ordinary PNG views and 42 of 144 ordinary JPEG75 views. The existing PNG sharp-boundary check localized 12 of 24 sharp edits and flagged 14 of 144 ordinary PNG views. The color-response candidate failed its frozen advancement rule in both formats. [Exact results](evaluation-panels/image-residual-color-transfer-2026-11-11.json)

## Why this comparison

Recent work studies noise-residual color channels for passive localization. Separate research on Photoshop inpainting motivates checking several frequency bands after JPEG delivery. This experiment uses its own fixed spatial bands and regional comparisons. It runs on the local CPU and uses no outside code, pretrained weights or source-camera attribution. [Residual color study](https://arxiv.org/abs/2608.20558) · [Multi-frequency JPEG study](https://doi.org/10.1109/TIP.2026.3699086)

At three scales, neighboring blurred versions are subtracted to isolate different ranges of fine detail. Each 16-pixel tile measures three cross-channel correlations and the share of variation in channel differences. Clipped or weak tiles are excluded. For each proposed rectangle, compare the supported inner tiles with four surrounding strips. The score is the third-largest core-to-strip difference minus the median strip-to-strip difference. One arm uses fine detail, one uses the first middle band, and one requires consistent direction of the change in both middle bands. The last arm keeps the smaller of the two responses at the same rectangle. A test with a deliberate green-channel residual reversal verifies that all three arms can recover the exact authored region.

## Sources and operating points

The same eight qualified development photos supplied six ordinary views per delivery. Each point is the second-highest of eight source-level control maxima or 0.015, whichever is larger. The prior evaluation rosters and the code digest were locked before development scoring. The fixed arm needed at least two additional useful feathered regions on each held panel with at most four additional ordinary flags on at most three source groups, assessed separately for each delivery. [Frozen method](evaluation-panels/image-residual-color-transfer-freeze-2026-11-11.json) · [Fitted points](evaluation-panels/image-residual-color-transfer-points-2026-11-11.json)

The two confirmation panels have distinct source images and credited photographers from each other and from development. Both had prior project exposure on other methods. They each supply 12 sharp and 12 feathered generated insertions and 72 ordinary processing or untouched controls in each format. The PNG and JPEG75 versions share the exact changed-pixel footprint before encoding. A useful location requires at least 0.30 intersection over union and at least 0.50 edited-pixel recall. The format references are the existing PNG sharp-seam comparison and the aligned JPEG double-compression region. Other image detectors were outside this comparison. [Earlier rights and edits](image-background-patch-transfer-2026-11-11.md) · [Later qualification](image-wavelet-region-transfer-2026-11-11.md)

| Panel | Delivery | Fine ordinary flags /72 | Mid ordinary flags /72 | Two-band ordinary flags /72 | Reference useful sharp /12 | Reference ordinary flags /72 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Earlier | PNG | 12 | 25 | 18 | 5 | 13 |
| Later | PNG | 18 | 24 | 14 | 7 | 1 |
| Earlier | JPEG75 | 12 | 12 | 18 | 0 | 0 |
| Later | JPEG75 | 6 | 18 | 24 | 0 | 0 |

All arms assessed every view. None selected a useful feathered location. The two-band arm selected the wrong region on 24 positive views. Before applying the point, its highest-ranked rectangle had useful geometry on one JPEG75 edit in the earlier panel and one PNG edit in the later panel, out of 96 positive deliveries. These retrospective counts indicate that changing the cutoff alone would leave most locations missed. The cue contributes no AI confidence.

## Reproduce

The CPU replay scored all 384 encoded views in 56 seconds. A separate Pillow calculation rebuilt all 1,536 selection masks, checked the exact encoded input and truth commitments, and reproduced the region counts. Exact source and truth files stay with the evaluator. Run from the repository root with the earlier and later qualified intakes and the retained earlier comparison journal.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze fit confirm; do
  backend/.venv/bin/python -B -m scripts.image_residual_color_transfer "$stage" \
    "$development" "$earlier" "$later" "$boundary_rows" "$output" || break
done
```
