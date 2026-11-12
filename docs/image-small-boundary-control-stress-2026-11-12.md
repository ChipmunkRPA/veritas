# Small image boundaries on photos and graphics

The fallback located 17 additional conventional pastes across 13 of 16 target photos. All 64 ordinary photo views kept their previous result. It added ten inspection boxes on 24 authored graphic views across seven of 12 graphics. The frozen test leaves unrestricted use experimental. [Audited source roster and case ledger](evaluation-panels/image-small-boundary-control-stress-2026-11-12.json)

| View | Count | Current check | With fallback |
| --- | ---: | ---: | ---: |
| Paste from another area of the same photo, correct location | 16 | 1 | 8 |
| Paste from a separate photo, correct location | 16 | 2 | 12 |
| Unchanged, tone-adjusted, blurred or resized photo, any box | 64 | 8 | 8 |
| Native authored graphic, any box | 12 | 3 | 10 |
| Resized authored graphic, any box | 12 | 7 | 10 |

The fallback preserved every incumbent crop and added zero incorrectly located boxes on the known pastes. Both paste types used exact native target masks. These conventional edits measure seam evidence and provide no information about the origin of the content inside a box.

## Frozen method and source roles

Reuse the published distributed-side score and fixed geometry. Keep the incumbent's box when it flags. Otherwise require a score strictly above 1.2805722063718288 and both sides between 56 pixels and 18% of their respective image dimensions. A useful box covers at least half of the exact changed pixels and reaches 0.30 intersection over union. Freeze the operating rule, source list, code hashes and acceptance criteria before scoring. Advancement required eight additional useful boxes across six targets, at least two in each paste type, at most two new incorrectly localized pastes, and at most three added ordinary views across two control groups. At most one unchanged photo group and one graphic group could gain a flag. [Original candidate and calibration](image-boundary-extent-transfer-2026-11-12.md)

Sixteen target photos and sixteen distinct donors come from the rights-qualified CC BY 2.0 v39 cache. Each target and donor used a different recorded source component and credited photographer. Earlier known trial roles and separately shortlisted geometric-copy rows were excluded using source identifiers and exact encoded and decoded hashes. A broad, unrelated copy proposal screen previously examined this cache. Its scores played no role in this selection. Eight targets were preselected by titles referring to features such as signs, buildings, vehicles or rooms. Other targets and donors used the frozen hash order. The source photographs do not have verified camera provenance. [Source inventory and rights](image-fresh-source-cache-v39-results.md)

Each target contributed a matched original, global tone change, feathered local blur, and resize round trip. One paste uses a different area in the same photo. The other takes pixels from a unique donor. The native target changes measure 64–85 pixels on one side and 64–86 on the other. The twelve separate authored graphic groups cover three layouts with variants across four image sizes. Dashboards, tile patterns and bar charts test whether a naturally sharp rectangle can trigger the fallback. Each graphic appears at its original size and after a matched resize round trip. These controls contain no third-party graphical assets and do not estimate the prevalence of this pattern in real customer uploads.

The graphic failures justify a separate prospective test for photo-only admission, including real screenshots and mixed photo–graphics. The trial does not establish that an admission rule can distinguish those inputs or that it preserves the paste gains. The earlier trial found zero of six independent generated inserts. The supplemental check remains a conventional boundary cue and has zero AI confidence weight.

## Verify and reproduce

The independent reader verified all 120 PNG deliveries, exact edited-pixel masks, and 79 arm-specific source crops. The no-media path recomputes the selection and integer localization outcomes from the public ledger. The test uses no model weights. No source photos, screenshots or derived crop pixels are distributed.

```sh
python3 -I -S backend/scripts/review_image_small_boundary_control_stress.py \
  docs/evaluation-panels/image-small-boundary-control-stress-2026-11-12.json
```

For native replay, save the public receipt's `freeze.roster` object as `$roster`. Place each exact attributed source in `$cache/<encoded_sha256>.bin`. Use a new `$run` directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_small_boundary_control_stress freeze "$cache" "$roster" "$run"
backend/.venv/bin/python -B -m scripts.image_small_boundary_control_stress run "$cache" "$roster" "$run"
backend/.venv/bin/python -B backend/scripts/review_image_small_boundary_control_stress.py "$run"
```
