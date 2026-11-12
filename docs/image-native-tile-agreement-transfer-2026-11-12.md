# Test local image edit evidence across two native tile layouts

The previous crop experiment established that smaller candidate regions can cover many more known edits. Its whole-crop response rarely ranked those locations first. This study tests a different response on 16-pixel native source cells and verifies whether it holds when the surrounding tile changes. The earlier one-third crop rule is an experimental comparator. [Previous crop study](image-crop-multiscale-transfer-2026-11-12.md)

## Frozen measurement

Each of two tilings covers the input with 384 × 384 tiles at source resolution. The second tiling starts 192 pixels earlier in each dimension. Reflection fills pixels outside the submitted bounds. The two early and middle feature layers are read on a common 16-pixel source grid. Each token is normalized by its channel norm and compared with the mean of the nearby cells in a seven-cell square ring around a three-cell square. The fit uses the resulting 768 values per cell. The checkpoint is the same pinned MIT-licensed image classifier used in the earlier comparison. [Pinned checkpoint](https://huggingface.co/OwensLab/commfor-model-384/tree/6076002bf0d9dd37537f965ee2f06f826c333b61)

The linear fit uses up to 24 cells with the highest exact feathered changed-pixel occupancy per source. Both native and feathered generated variants provide positive training examples. The same locations in aligned ordinary processing and separate remote cells from the edited views provide negative examples. The two classes contribute equally within each source, and all sources carry equal weight. Recent work motivates measuring local cues in the presence of processing outside an edit. This is an independent small linear fit using the already licensed checkpoint and archived media. [November inpainting study](https://arxiv.org/html/2602.00192v2) · [November regional localization study](https://arxiv.org/abs/2609.36882v3)

The single-context arm uses the first layout. The agreement arm takes the lower standardized response at each source cell across both layouts. Both propose one source rectangle from a fixed set of side lengths of three to sixteen cells and aspect ratios up to two. The point for each arm strictly exceeds the second-highest ordinary calibration group maximum. The previous research comparator uses its two established one-third crop points. The measured scope is upright RGB PNG with a short side of at least 384 and a long side of at most 768 pixels. The candidate receives only the submitted pixels during evaluation. A selected area is a review cue with zero weight in AI authorship confidence.

A useful location requires 0.30 native changed-pixel intersection over union and 0.50 recall. The frozen gate requires two added primary feathered locations on different source groups, one added alternate-editor or screenshot feathered location, at most one lost comparator location, at most two newly flagged ordinary photo groups, and at most one newly flagged ordinary interface group.

## Source roles

The fit uses the 12 previously qualified backup LaMa groups and seven separately qualified source-preserving SDXL and Kandinsky groups. Eight credited Commons contributors who were outside both earlier crop fitting panels provide 32 original, tone, blur and resize views for calibration. The primary held role uses the 12 retained v225 LaMa groups that were outside the previous multi-scale panel. The second held role uses three separate SDXL and three Kandinsky source groups from the archived later qualification. They contribute independent original, global processing, local processing, native and feathered views. The photos have prior project exposure. This study keeps the fit, calibration and held roles disjoint by recorded identity and exact RGB, and checks 794 cross-role thumbnail pairs for near matches. [Primary and backup LaMa rights](image-bg-real-backup-intake-v262-method.md) · [Alternate generator qualification](image-wavelet-region-transfer-2026-11-11.md) · [Commons photo credits and terms](image-affine-native-scale-2026-11-11.md)

A separate held view embeds each of the 12 primary feathered images in a native Chromium screenshot. Each source also has the matching ordinary photo preview and a text-layout control. Four authored templates are reused with new photos and positions. The photo pixels agree exactly with the underlying crop and the matched screenshot differences remain within that crop. The primary photos and their screenshot variants share source groups. The frozen panel includes 107 training views, 32 calibration views and 138 held views.

## Held results

Neither tile arm found a useful location on the 48 known edited views. Even before the operating point, neither arm's highest-ranked rectangle reached the required overlap on any of the edited views. The earlier research crop rule located one of the six native alternate-editor edits. The new arms missed that location. [Audited scalar ledger](evaluation-panels/image-native-tile-agreement-transfer-2026-11-12.json)

| Known edited views | Earlier flags | Earlier useful | Single-context flags | Single-context useful | Agreement flags | Agreement useful |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary native edits | 4/12 | 0/12 | 4/12 | 0/12 | 2/12 | 0/12 |
| Primary feathered edits | 2/12 | 0/12 | 2/12 | 0/12 | 1/12 | 0/12 |
| Alternate native edits | 3/6 | 1/6 | 1/6 | 0/6 | 0/6 | 0/6 |
| Alternate feathered edits | 1/6 | 0/6 | 1/6 | 0/6 | 0/6 | 0/6 |
| Feathered edits within screenshots | 3/12 | 0/12 | 12/12 | 0/12 | 11/12 | 0/12 |

| Ordinary views | Earlier flags | Single-context flags | Agreement flags |
| --- | ---: | ---: | ---: |
| Primary original | 1/12 | 3/12 | 2/12 |
| Primary global tone | 1/12 | 3/12 | 2/12 |
| Primary local blur | 2/12 | 3/12 | 2/12 |
| Primary resize | 1/12 | 3/12 | 2/12 |
| Alternate original | 0/6 | 1/6 | 0/6 |
| Alternate global processing | 0/6 | 1/6 | 0/6 |
| Alternate feathered processing | 3/6 | 1/6 | 0/6 |
| Total ordinary photo views | 8/66 | 15/66 | 8/66 |
| Ordinary text-layout screenshots | 0/12 | 9/12 | 7/12 |
| Ordinary screenshots with a genuine photo | 3/12 | 12/12 | 9/12 |
| Total ordinary screenshot views | 3/24 | 21/24 | 16/24 |

The agreement check reduced the ordinary screenshot flags from 21 to 16 compared with one context. It still newly flagged nine ordinary screenshot groups compared with the earlier crop rule, exceeding the frozen limit of one. The ordinary photo group totals were five, four and two respectively, with two newly flagged agreement groups. The model's strongest regions remained unsuitable for useful localization at either point. This result keeps both tile arms experimental and the app continues using the existing image checks.

The fit used 1,736 positive and 4,960 negative source-cell observations across both contexts. Fit and calibration ran in 45 seconds with 1,109 actual native tile forwards and 440 exact reuses. The held run completed in 146 seconds with 1,020 actual native tile forwards and 501 exact reuses. The earlier crop rule used 2,246 actual held crop forwards and 1,204 exact reuses. The held worker reached 0.90 GiB peak RSS. The prospectively sealed benchmark digest is `2eacbfb5f2388c55560215e23430d818025268ee3d2e111cd18fa085698a9785`.

The separate reader verified all 277 views, 3,070 reflected tile inputs, 19 training-cell rosters, 340 scored spatial maps, 414 held region overlaps and 3,450 historical-method crop inputs. The source-role review covers this local fit. Upstream checkpoint training membership and unrecorded photographer aliases remain unknown.

## Reproduce

The public reader independently recomputes the two calibration points and every review decision and aggregate. The native readback reconstructs each reflected source tile, the exact fit-cell choices, the two-context score fusion, the spatial maxima, the historical baseline crops and the changed-pixel overlap from retained masks. It uses stored model scores. The public receipt omits photos, screenshots, masks, individual input hashes, fitted weights and dense score maps.

```sh
python3 -I -S backend/scripts/review_image_native_tile_agreement_transfer.py \
  docs/evaluation-panels/image-native-tile-agreement-transfer-2026-11-12.json
```

Use the retained Commons archive, primary v225 and backup v262 intakes, seven-group generator intake, later wavelet qualification, preceding crop-layout run, remote-crop comparison and pinned checkpoint. Run the offline Chromium capture before sealing the benchmark.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.image_native_tile_agreement_transfer prepare \
  "$commons" "$primary" "$backup" "$legacy" "$alternate" "$previous" "$baseline" "$weight" "$run"
node backend/scripts/capture_image_crop_multiscale_transfer.cjs "$run"
for stage in freeze fit held; do
  backend/.venv/bin/python -B -m scripts.image_native_tile_agreement_transfer "$stage" \
    "$commons" "$primary" "$backup" "$legacy" "$alternate" "$previous" "$baseline" "$weight" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_native_tile_agreement_transfer "$run/public.json" --native \
  "$commons" "$primary" "$backup" "$legacy" "$alternate" "$previous" "$baseline" "$weight" "$run"
```
