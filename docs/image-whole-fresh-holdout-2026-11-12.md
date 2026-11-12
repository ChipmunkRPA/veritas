# Whole-image generation on a separate source panel

The supplementary check flagged 18 of 18 fully generated images in each delivery format and zero of 16 ordinary photo sources in either. It added 12 native and 15 JPEG75 indications to the paired Full analysis. The supplement contributes zero weight to the calibrated confidence and displays its inspected image inputs. [Source and delivery freeze](evaluation-panels/image-whole-fresh-sources-2026-11-12.json) · [Paired comparison](evaluation-panels/image-whole-fresh-comparison-2026-11-12.json) · [Earlier development result](image-smaug-v7-development-evaluation.md)

## Sources and scope

The generated roster has six exact publisher JPEGs each from Qwen-Image-2512, GLM-Image and FLUX.2 Max. The Qwen-Image-Bench publisher declares Apache 2.0 for the exact dataset revision. The frozen selection ranks prompt IDs 1–120 and gives each selected prompt to only one generator. The ordinary roster has 16 JPEG files from 16 separately credited Wikimedia Commons photographers. They come from the previously reviewed 38-source Commons archive. Fifteen use the publisher's retained thumbnail delivery and one uses the retained original-derived delivery. Their individual CC BY and CC BY-SA terms and attribution URLs are in the public source receipt. The ordinary labels use publisher descriptions and the earlier source review. Camera capture provenance is unverified. [Pinned benchmark card](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench/blob/d2493deb153b020cf169c7e3f57d15e4dd697038/README.md) · [Commons intake and source terms](image-affine-native-scale-2026-11-11.md)

The earlier camera-category intake returned no eligible listings under its original file and attribution limits. The replacement used the existing attributed archive and kept every generated selection unchanged. Selection and rights review occurred before detector scoring. All 34 sources pass exact encoded-byte, oriented-RGB and four-bit perceptual-hash overlap checks against the earlier development inventory and one another. Earlier project use of the ordinary photographs, transformed duplicates and candidate pretraining overlap remain possible. The ordinary photo scenes and the generated prompts represent different source distributions.

Each source has one exact retained native JPEG and one upright RGB JPEG75 delivery with 4:2:0 subsampling and no EXIF. The two formats are paired views of the same 34 groups. The previously published strict positive operating point is unchanged. Candidate inference uses the exact Apache-2.0 detector and base weights, CPU float32, and the same three pinned 528-pixel crops. The strongest claim from the candidate's output is an uncalibrated whole-image cue. [Detector and base terms](image-smaug-v7-preparation.md) · [Previously frozen point](evaluation-panels/image-v7-smaug-genuine-operating-point.json)

The paired trial compares the current Full image conclusion with Full plus a positive candidate cue. It counts statistical AI directions separately from signed provenance, unavailable results and inconclusive assessments. The frozen advance rule requires all 68 responses in both arms, at least two additional generated-image indications from two families per format, and at most one additional ordinary group flagged per format. The 18/18 and 0/16 candidate counts do not establish a general detection or false-alarm rate.

## Paired outcomes

| Delivery | Full AI indications on generated images | With the supplementary cue | Added generated indications | Added ordinary-photo flags |
| --- | ---: | ---: | ---: | ---: |
| Native JPEG | 6/18 | 18/18 | 12 | 0/16 |
| JPEG75 | 3/18 | 18/18 | 15 | 0/16 |

The additions cover all three generator families in each format and meet the frozen gate. Full originally returned 10 inconclusive and 2 original-leaning component assessments for native generated files, and 11 inconclusive, 3 original-leaning and 1 unavailable assessment for the JPEG75 files. The five original-leaning results would conflict with the positive cue. Full found zero AI indications in the ordinary controls, where 31 of 32 deliveries were unavailable and one was inconclusive. A low candidate response supplies no original-file evidence.

## Local execution and review

The candidate completed all 68 deliveries in 153 seconds with 4.07 GB peak sampled process-tree RSS. Its native loader verified every pinned base and head value. The run recorded three finite crop responses per delivery and reproduced the reductions with independent double-precision arithmetic. A separate source replay regenerated all 204 crop pixel hashes and rational source coordinates. The public reader rejected a changed license, a duplicate source and an altered delivery freeze.

The Full arm ran the tracked app at commit `4119adc1c8332d6226d066638818a751d5d6ce1c` with 19 enabled checks, exact local settings and verified approved weights. It completed all 68 analyses in 665 seconds with 3.78 GB peak sampled process-tree RSS. Across 1,292 check statuses, 966 were successful and 326 were outside scope. Both model processes exited successfully, and the independent reader recomputed all paired outcomes. The first Full bootstrap stopped before scoring when the snapshot lacked migrations. Restoring the tracked migrations from the same commit allowed the unchanged panel to complete.

The public reader verifies the source-freeze commitment, exact paired delivery membership, terms, and prior-source exclusion without third-party packages. Supplying the local archive also verifies every delivered file hash. The native runner requires the existing research environment and the separately acquired, exact pinned weights.

```sh
python3 -I -S backend/scripts/review_image_whole_fresh.py \
  --results docs/evaluation-panels/image-whole-fresh-comparison-2026-11-12.json

export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
for stage in plan acquire freeze; do
  backend/.venv/bin/python -B -m scripts.image_whole_fresh_sources "$stage" "$trial" || exit 1
done
tmp/image-smaug-v7-env/bin/python -B -m scripts.run_image_whole_fresh candidate "$trial"
backend/.venv/bin/python -B backend/scripts/review_image_whole_fresh.py \
  --directory "$trial" --candidate-journal "$trial/candidate.jsonl"
```

For the Full arm, put a tracked copy of `backend/app` and `backend/alembic` under `$trial/checkout`, record its commit in `$trial/checkout/commit.txt`, and prepend `$trial/checkout/backend` to `PYTHONPATH`. Run the `freeze-baseline` and `baseline` arms with the app environment. The freeze records the current local Full settings, policy, model hashes, and exact code snapshot. The runner uses an isolated local store and retains every case's stage and decision records. Source images and weights stay outside Git.

## Local app configuration

Set `VERITAS_IMAGE_WHOLE_REVIEW_PYTHON` to the absolute `bin/python` path of the pinned research environment to enable the check for new Full image analyses in development. `VERITAS_MODELS_DIR` points to the root containing `research/image-v7-smaug`. The check covers upright RGB JPEGs with a short side of at least 1,024 pixels and a long side of at most 4,920 pixels. Other formats show the scope limit. The separate worker validates the pinned configuration and weights, runs offline, and retains the exact inspected crops only after a positive cue. When available host memory is low, it can release unused model allocations through the existing cache manager and recheck capacity before launching. A conflicting calibrated original-leaning component remains visible under an inconclusive overall headline. Source rectangles show which pixels each inspected view came from.

A separate native PNG trial flagged all 16 generated files and six of eight ordinary browser captures. The control result kept broad PNG support outside the app. [PNG coverage test](image-whole-png-transfer-2026-11-12.md)

Two Full app runs on the Mac used simulated checkout and the local Metal configuration. The generated fixture published three exact 528 × 528 views whose API payload hashes matched JSON, and whose pixels matched the Standard and Summary PDFs. Chrome loaded all three views in the plain result. The calibrated component was preserved and the customer headline became “Inconclusive: image checks disagree.” The ordinary fixture produced a measured no-flag record, no inspected-view exhibits, and kept its existing inconclusive headline. Both reports and the customer API projections used the same evidence wording without model names.
