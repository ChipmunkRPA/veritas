# Locate small generated edits in a single JPEG

Four new region-ranking methods found no useful additional locations on
336 held edit deliveries. Lowering the internal review point would leave
the highest-ranked region outside the required overlap on every edit.
Chipmunk Verify keeps the existing image checks.

## What changed in the method

Earlier patch checks could respond at known edit coordinates and still
select a different region in a single uploaded JPEG. Recent research
also found that image-wide scores can depend on artifacts outside an
inpainted region. This experiment retains the original surroundings and
trains the region selector directly on differences between correctly
located edit sites and decoys. It uses an independently implemented
pipeline and a commercial-use Apache-2.0 patch encoder. The exact
configuration and checkpoint digests are recorded in the comparison.
[Inpainting study](https://arxiv.org/html/2602.00192v2) ·
[Pinned patch encoder](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)

The check reads 14 × 14 native cells using two intermediate encoder
layers on overlapping 224-pixel tiles. A separate extractor measures
luminance and color residual energy at four spatial scales. Each region
uses an inside measurement and its difference from a surrounding ring.
The fit compares the true site with other regions in the same photo and
the same site in the original, locally blurred, toned and copied views.
The last variant also learns from the highest-scoring wrong locations
found in fitting. The four compared methods use the frozen visual
features, the residual features, both together, and both with this extra
negative sampling. Each check has its own weights and operating points
for quality 75 and quality 90.

## Sources and overlap

Twenty-four TGIF and TGIF2 photo groups supplied fitting data. A further
24 groups set the operating points and chose the feature combination.
The held comparison used 48 other TGIF or TGIF2 sources with Firefly,
Stable Diffusion 2 and FLUX Fill edits. A separate BG-REAL panel used 36
photo groups with LaMa object removals. The source pictures had prior
project exposure in other studies. This comparison separates original
byte and RGB hashes across its roles. Identity aliases and pretrained
exposure are unknown. [TGIF and TGIF2 source terms](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea)
· [BG-REAL source terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

The paired JPEGs use the same 4:2:0 delivery recipe. Each source
produces the published local edit, an eight-pixel inward blend, an
original, and three matched ordinary retouches. The known edit region
contains exactly the changed pixels before JPEG encoding. A useful
reported rectangle requires at least 0.30 intersection over union
and at least 50% recall on those pixels. The evaluation reports capture
and editing effects in these staged deliveries. The two encodings and
multiple edits from one photo share the same source group.

## Measured results

The residual-only method was selected before held scoring at both
qualities. The stricter setting allowed zero new ordinary photo groups
at selection. The wider setting allowed two. The current comparator
uses the existing luminance compression check and the regional check
where its local serving profile applies. The luminance check assessed all
1,008 held deliveries. The optional regional check assessed 198 BG-REAL
JPEG75 deliveries within its source profile.

| Source panel | Quality | Useful locations, current → with wider candidate | Ordinary flags, current → with wider candidate |
| --- | --- | ---: | ---: |
| TGIF/TGIF2 held | 75 | 0 → 0 of 96 | 0 → 16 of 192 |
| TGIF/TGIF2 held | 90 | 4 → 4 of 96 | 9 → 30 of 192 |
| BG-REAL transfer | 75 | 0 → 0 of 72 | 15 → 34 of 144 |
| BG-REAL transfer | 90 | 0 → 0 of 72 | 0 → 16 of 144 |

The wider candidate added 72 ordinary flags and 39 misplaced edited-file
highlights. The stricter version added zero alerts. The native search
included a region meeting the overlap rule on 332 of 336 edited
deliveries. In all 332, every such region scored below the wider operating
point. On the remaining four, the fixed search geometry fell short of the
overlap rule. The top-ranked region missed the overlap rule in all four
methods on every edited delivery before applying any operating point.
These results give no basis for using any of the four as a new check.
[Per-delivery results](evaluation-panels/image-region-ranking-2026-11-16.json.gz)

## Independent readback

A separate pass checked 1,584 encoded-file hashes, 792 native truth
masks and 4,032 candidate intersections. A 512-bit thumbnail-gradient
check found no close matches in 6,336 cross-role photo comparisons.
The public receipt includes the source-group commitments, scalar
scores, output rectangles and overlap counts. Source media, exact
truth masks and intermediate visual tensors stay in the local workspace.

## Replay and compare another check

Use the standard-library result reader to recompute the operating-point
choices and outcome counts. To rebuild inputs, use the exact TGIF
archives listed in the [earlier source preparation](image-jpeg-signed-transitions-2026-11-12.md#local-replay)
and the two local qualification receipts and media directories from the
[BG-REAL source intake](image-residual-template-transfer-2026-11-13.md#reproduce).
Use the pinned BG-REAL `manifest_v2.csv` with those qualified sources.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/image-region-ranking-2026-11-16.json.gz
python3 -B -S -m scripts.image_region_rank_compare verify "$receipt"

runner=backend/.venv/bin/python
"$runner" -B -m scripts.image_region_rank_inputs "$tgif" "$run/deliveries" \
  --publisher-index "$bg_manifest" \
  --transfer "$qualification_v225" "$media_v225" \
  --transfer "$qualification_v262" "$media_v262"
"$runner" -B -m scripts.export_image_region_rank_review "$receipt" "$run/deliveries" "$run/review"
```

The exporter creates local manifests, private binary truth masks and
example responses in the same format used for the project-wide review.
Each delivery quality has 144 development and 504 confirmation inputs.
Use `relative-deliveries.json` to locate each exact JPEG. Record your
detector's original input hashes and reported regions, set its own
operating point on development, then compare it on confirmation. The
standard-library `count-locations` and `summarize` commands reproduced
both generated aggregates from the original masks. The aggregate output
is suitable for sharing after checking its labels. [Independent review format](independent-testing.md#count-exact-locations-on-your-own-files)

To reproduce the fitted variants, calculate the current checks and the
fit and calibration feature banks before running `fit`. Use the same
encoder revision for all four feature banks. The recorded run uses Apple
Metal float32. `assess` requires the frozen readout and fitting code and
the complete current-check journal.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
"$runner" -B -m scripts.image_region_rank_reference "$run/deliveries" "$run/reference.jsonl"
"$runner" -B -m scripts.image_region_rank_features "$run/deliveries" "$run/rank-fit" fit
"$runner" -B -m scripts.image_region_rank_features "$run/deliveries" "$run/rank-choice" calibration
"$runner" -B -m scripts.image_region_rank_compare fit "$run"
for role in held transfer; do
  "$runner" -B -m scripts.image_region_rank_features "$run/deliveries" "$run/rank-$role" "$role" || exit 1
done
"$runner" -B -m scripts.image_region_rank_compare assess "$run"
"$runner" -B -m scripts.audit_image_region_rank inputs "$run/deliveries" "$run/input-audit.json"
"$runner" -B -m scripts.audit_image_region_rank release "$run" "$run/result.json.gz"
```
