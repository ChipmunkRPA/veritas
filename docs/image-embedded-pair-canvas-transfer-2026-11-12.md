# Exact image comparison across screenshot sizes

The canvas transfer reproduced useful exact pixel comparisons in all five sizes. Each size also reported some comparisons when a resized third view of the photo was present. That failure includes 768 × 768, a size already covered by the local app. The trial supplied no basis to expand the serving sizes. [Frozen outcomes](evaluation-panels/image-embedded-pair-canvas-transfer-2026-11-12.json) · [Earlier third-view study and app scope](image-embedded-pair-third-guard-2026-11-12.md)

## Frozen panel

The study used 16 attributed photo groups with known native edits and six further photos as standalone controls. Six edits used previously qualified FLUX composites, four came from qualified LaMa sources unused in the prior paired-image test, and six were ordinary copy-and-paste changes to photographs. It also captured eight separately authored ordinary interfaces in Chromium at each size, including a layout with repeated previews. The author and photo checks prevent source identity overlap within the panel. The licensed image banks had prior project exposure. No new generator ran. [FLUX source qualification](image-flux-photo-expansion-v285-result.md) · [LaMa source qualification](image-bg-real-backup-intake-v262-result.md) · [Photo source and rights checks](image-fresh-source-cache-v39-results.md)

The exact submitted RGB values and edit masks were frozen before measurement. Each size has 174 inputs, including 48 eligible two-pane edit cases, 46 ordinary controls, and 80 third-view cases. The third views were an exact full copy, an exact partial copy, and copies resized to 0.64×, 0.80×, and 1.18×. Pixel phase and pane placement vary with the canvas. The runner uses the current matching and verification code at every size, including sizes the app currently skips. Its isolated scoring processes measured native resource use. The frozen study required minimum useful coverage, exact masks, zero ordinary or extra-copy reports, a minimum count of challenges reaching the third-view check in each category, and peak process RSS no higher than 512 MiB.

## Results

| Canvas | Exact sharp edits | Exact feathered edits | Exact ordinary local edits | Reports with resized third view | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 768 × 768, current scope | 10/16 | 10/16 | 16/16 | 7/48 | 269 MiB |
| 1024 × 768 | 9/16 | 9/16 | 16/16 | 7/48 | 319 MiB |
| 1280 × 720 | 10/16 | 10/16 | 16/16 | 8/48 | 355 MiB |
| 1366 × 768 | 10/16 | 10/16 | 16/16 | 7/48 | 370 MiB |
| 1920 × 1080 | 10/16 | 10/16 | 16/16 | 8/48 | 703 MiB |

All 230 ordinary controls produced zero reports. All measured masks on the valid two-pane edits were exact. The four larger sizes preserved 34, 36, 36, and 36 of the 36 useful comparisons in the 768 × 768 lane, with zero new groups. The same photo groups appear at each size.

Across sizes, the check reported 21 of 80 cases with a third copy at 0.64×, 10 of 80 at 0.80×, and 6 of 80 at 1.18×. The exact partial and full thirds were all withheld. All exact full thirds were already rejected by the initial pair-selection step, so the dedicated guard received zero cases of that type. The 1920 × 1080 lane also exceeded the memory budget. No size met the frozen gate. The measured weakness is ambiguous pairing when a third copy was resized. A reported pixel difference remains a difference between those aligned areas and gives no AI-authorship evidence.

The native readback reassembled all 870 views and verified 63,696 exact texture witnesses. The separate standard-library reader checked the published results and each rollout criterion. The public package excludes the image pixels, private input hashes, and individual texture coordinates. The original freeze SHA-256 is `a67a49ce853fcdc765ea5f8b4e03d53f129992572ff621940b3f8a73a47af955`.

## Reproduce

Use the retained photo archives, the pinned detector code in the receipt, and the original browser configuration. Set `run` to a new local directory. For exact screenshot replication, use the same Chromium version and system fonts as the recorded capture. Keep the 40 PNG controls and all input images local.

```sh
mkdir -m 700 "$run"
node backend/scripts/capture_image_embedded_pair_canvas_controls.cjs "$run"
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_embedded_pair_canvas_transfer freeze "$run"
for size in 768x768 1024x768 1280x720 1366x768 1920x1080; do
  backend/.venv/bin/python -B -m scripts.image_embedded_pair_canvas_transfer score "$run" --size "$size" || exit 1
done
backend/.venv/bin/python -B -m scripts.image_embedded_pair_canvas_transfer audit "$run"
```

```sh
python3 -I -S backend/scripts/review_image_embedded_pair_canvas_transfer.py \
  docs/evaluation-panels/image-embedded-pair-canvas-transfer-2026-11-12.json
```
