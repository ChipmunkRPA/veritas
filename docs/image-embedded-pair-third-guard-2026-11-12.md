# Detect a third matching view before reporting a two-pane difference

The unchanged single-file comparison found real differences in all 12 ambiguous three-pane cases in the preceding study. The new guards reject an apparent two-area comparison when a separately placed third view has strong matching support. The guarded comparison passed its separately frozen holdout. It retains exact two-sided pixel evidence and gives it zero AI authorship weight. [Earlier confirmation and failure](image-embedded-pair-difference-confirmation-2026-11-12.md) · [Audited scalar results](evaluation-panels/image-embedded-pair-third-guard-2026-11-12.json)

## Frozen method and sources

Keep the exact two-pane comparison unchanged. The first guard seeks a third native translation for the same exact 16 × 16 RGB witnesses. It requires at least 12 tiles across six cells, a native span of at least 128 pixels in each dimension, and separation from the two reported areas. The second guard seeks a geometrically consistent third view with local features. It requires at least 12 geometric inliers, a sufficient spatial span and agreement on at least ten separately positioned local pixel patches after alignment. This second guard can abstain when a resized third view changes the native pixel values. Only the original exact two-area comparison supplies any reported pixel difference.

The held panel uses 12 qualified v225 source groups separate from the earlier two-pane confirmation and four separate qualified photos as unrelated third views. Native RGB photo panes are authored in independently frozen horizontal and diagonal layouts. Each group contributes a real sharp and feathered edit, an ordinary local change, a true pair beside an unrelated third photo, a full exact third copy, a 152-pixel-wide partial third, a third view resized to about 78%, and an unchanged duplicate. Both matched edited areas preserve the exact displayed difference. The measured PNG geometries are 1,088 × 688 and 944 × 944. [Qualified archive and exact masks](image-bg-real-transport-v225-result.md)

The preregistered gate preserves every exact valid two-area mask and suppresses all admitted exact and partial third-copy reports. It requires adequate coverage of each ambiguity stratum and suppression of at least 75% of the admitted resampled third-copy views. Duplicate comparisons must report zero differences. The guarded method never adds a difference outside the original two-area comparison.

## Held results

| Held condition | Unguarded reported differences | After both guards | Exact valid masks retained |
| --- | ---: | ---: | ---: |
| Sharp known edit | 9/12 | 9/12 | 9/12 |
| Feathered known edit | 10/12 | 10/12 | 10/12 |
| Ordinary local change | 12/12 | 12/12 | 12/12 |
| Valid pair beside unrelated third photo | 12/12 | 12/12 | 12/12 |
| Full matching third | 8/12 | 0/12 | — |
| Partial matching third | 12/12 | 0/12 | — |
| Resampled matching third | 12/12 | 0/12 | — |
| Identical duplicate | 0/12 | 0/12 | — |

The exact guard found three of eight admitted full thirds and all 12 admitted partial thirds. The geometric guard found all eight admitted full thirds, one partial third and all 12 resampled thirds. All 43 exact valid two-area masks stayed intact. The existing proof already abstained on four of the 12 full three-view layouts. Both guards together removed every remaining three-view report. Two guard outputs can agree on the same case.

A separate position check matched all 12 partial third translations to the authored pixel offsets. Across the 12 resized views, fitted scales were 0.77981–0.78050 and the maximum error at the source-pane center was 0.18 pixels against the authored resized placement. This study covers the stated placements, the measured resize factor and the exact source-domain limits. It supports a bounded native comparison cue with separate, neutral handling for an additional matching view.

The 18-second CPU run covered 96 native views. Readback verified 4,555 exact third-copy tiles, 2,693 geometric verification patches and 75 exact original comparison masks, including the 32 later withheld. The frozen seal is `e7a269527e82777422b3e4450daea75fc8edfecf167fa1b2d1085d2ad8c339e6`. The earlier failed pilot and confirmation stay unchanged.

## Local app

Set `VERITAS_IMAGE_EMBEDDED_PAIR_REVIEW=true` in a development instance to show **Pixel differences between matching areas** on new image analyses. It covers single-frame, upright RGB PNGs at four tested canvas sizes: 768 × 640, 768 × 768, 1,088 × 688, and 944 × 944. The guarded production implementation reproduced all 96 held outcomes and 43 exact masks. Replaying the two earlier canvas sizes preserved 44 valid pairs and withheld all 12 ambiguous three-view comparisons.

The ordinary view toggles both crops between original pixels and highlighted differences. The two links open the measured coordinates on the submitted image. The PDF includes the four plates and their checksums. The JSON and signed package also retain the exact binary mask, source coordinates and full witness record. The signed source check verifies the original PNG, witness tiles, compared crops and mask. This comparison carries zero weight in the AI authorship score.

Two actual local checkout and worker runs used simulated payment. The supported example showed **2,503** differing pixels and reproduced all five source assets. The three-view control withheld the comparison. Both results agreed across API, PDF, JSON and signed source verification. Chrome loaded the four plates in the ordinary view and verified both source links. [Local app receipt](evaluation-panels/image-embedded-pair-app-2026-11-12.json)

## Reproduce

```sh
python3 -I -S backend/scripts/image_embedded_pair_third_guard.py review \
  docs/evaluation-panels/image-embedded-pair-third-guard-2026-11-12.json
```

Use the retained primary, earlier generated and backup archives, the pilot generated archive, and the earlier confirmation run directory for the exact role and code commitments.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze run audit; do
  backend/.venv/bin/python -B -m scripts.image_embedded_pair_third_guard "$stage" \
    "$primary" "$earlier" "$backup" "$pilot_generated" "$confirmation" "$run" || exit 1
done
```
