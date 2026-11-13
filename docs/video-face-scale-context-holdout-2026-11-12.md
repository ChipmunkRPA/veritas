# Scale-only context on a separate video panel

The previous context study found that a combined zoom and brightness correction could erase genuine speaker changes. Scale alone removed all five digital punch-in annotations in that earlier panel while retaining the observed speaker changes. This study fixes the scale-only rule and tests whether it works on different source groups and on the app's own selected ranges. [Earlier calibration and ablations](video-face-context-correction-2026-11-12.md)

## Frozen comparison

Ten distinct CREMA-D originals, with actor codes outside the prior near-head study, form five pairs. The fixed roster prefers previously qualified selection-role recordings with at least 55 source pictures, then uses the same eligibility in the prior confidence-calibration role. The clips had earlier project exposure. All retain native source commitments. Each pair supplies a fixed-layout genuine speaker cutaway, a digital punch-in, a local gain step and an unchanged view. The cutaway and punch-in are also tandem-compressed. The authored intervention is sixteen or eighteen frames in an exact four-second 720p30 delivery. [Source preparation and license](video-face-near-head-transfer-2026-11-12.md)

The candidate uses the existing held comparison's scale-only fit and the sealed acceptance point of three levels and 45% improvement on held pixels, with a two-level minimum improvement in the center. It considers only ranges already showing broad change at both edges. Each fitted scale must move at least 7% from unity, with opposite directions at entry and exit. The product of the scales must remain within 0.07 of one. The annotation would say that scale alignment reduced the observed change. It would preserve the existing face range and carry zero AI-confidence weight.

The same frozen candidate runs in two arms. The source-timed arm uses exact authored edges. The application arm uses the first recorded brief, refined or coarse range from the actual Full worker. It looks for central transitions within eight native frames of those endpoints before applying the scale fit. The candidate is assessed only where the existing worker reports a range. The Full route runs the existing tracking and appearance detectors in a scratch store with no customer database or billing. The study gate requires at least three app-selected zoom ranges and at least two useful scale explanations, at least six selected genuine cutaway ranges, at least 75% preservation of those genuine cutaways, and no wrong explanations. The authored arm requires seven zoom explanations, four retained genuine source pairs and at most one wrong explanation. The protocol, source roster and gate were sealed before measuring either arm.

## Results

| Delivery | Files | Broad at authored edges, raw → corrected | Scale explanations at authored edges | App-selected ranges | Scale explanations at app ranges |
| --- | ---: | ---: | ---: | ---: | ---: |
| Genuine speaker cutaway | 5 | 5 → 5 | 0 | 5 | 0 |
| Tandem genuine cutaway | 5 | 5 → 5 | 0 | 5 | 0 |
| Digital punch-in | 5 | 5 → 0 | 5 | 0 | 0 |
| Tandem digital punch-in | 5 | 5 → 0 | 5 | 0 | 0 |
| Local brightness step | 5 | 2 → 2 | 0 | 0 | 0 |
| Unchanged | 5 | 0 → 0 | 0 | 0 | 0 |

All ten genuine cutaways retained broad context when the existing app supplied the interval. Each selected native edge was within two pictures of its exact authored boundary. All ten zoom deliveries qualified when given exact timing, with no wrong scale explanations on the remaining twenty deliveries. The existing app selected no zoom range, so there was no eligible case to receive the new explanation in the app. This passes the source-timed gate and fails the serving gate. The scale-only qualifier stays experimental. A useful confirmation would need zoom or reframing changes that actually receive an appearance range, with source commitments and controls fixed before testing. [Held counts and native audit](evaluation-panels/video-face-scale-context-holdout-2026-11-12.json)

The initial preparation reached the Full worker's isolation guard before any inference. The setup imported settings too early. The next freeze corrected import order and preserved the exact roster and scoring rule. All thirty delivered files and native frame commitments matched the first preparation. The Full worker took 180.1 seconds, and the context pass took 14.6 seconds on one local worker.

## Replay

Use a new private working directory and the retained source archives. The audit rebuilds each delivery and verifies source PTS and RGB commitments for every selected boundary. Private per-source fits and native pictures stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze prepare full measure report; do
  backend/.venv/bin/python -B -m scripts.video_face_scale_context_holdout "$phase" "$run" || exit 1
done
```
