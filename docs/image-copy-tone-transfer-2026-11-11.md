# Geometric copy comparison after a local tone change

The tone-corrected comparison found two additional sharp PNG copy locations and one additional sharp JPEG75 copy location. It recovered no useful feathered locations. No ordinary view gained a flag. Neither format met the frozen requirement of gains on three distinct tone-adjusted source photos. The added comparison remains offline. [Audited results and public photo credits](evaluation-panels/image-copy-tone-transfer-2026-11-11.json)

## Comparison

Use the existing same-image geometric copy detector and keep every dense mask it produces. For each of its first three distinct sparse proposals without a dense mask, align the second area to the first. Among textured 11-by-11 neighborhoods inside the sparse seed with luma correlation of at least 0.96, fit a single gain and offset from the median local standard-deviation ratio and the corresponding mean difference. Require at least 100 seed pixels, gain from 0.75 to 1.35, and absolute offset at most 38 levels. Compare the corrected luma values, retaining a connected region supported by the seed. Map its mask back onto the second area. This test uses no trained weights. Prior work motivates evaluating naturally similar objects independently of deliberate copies. [Copy-move false-alarm research](https://arxiv.org/abs/1906.00649) · [Similar genuine objects and benchmark terms](https://github.com/wenbihan/coverage)

The fixed panel contains eight rights-qualified originals from a retained insertion study and four separately acquired photographs with repeated windows or docked bicycles. The 12 photos have distinct credited photographers and encoded source digests. Four other planned public photographs were unavailable at intake and produced no measurements. The test ran on the available frozen sources. It used only the public photos and retained originals identified in the receipt. [Retained source qualification](image-bg-real-insertions-2026-11-11.md)

Each source supplies sharp and feathered rotated copies, both with and without a tone adjustment of the copied pixels. The unchanged evaluation source, a whole-image tone adjustment, a local tone adjustment and a resize round trip serve as controls. The public source photographs are oriented and resized before the controlled interventions. A useful pair must have at least 0.30 intersection over union and 0.50 recall on both exact pre-encoding masks. The second arm must gain useful tone-adjusted pairs on three distinct photos per format, preserve the incumbent pairs, and add at most one ordinary pair with none on unchanged evaluation sources.

## Results

| Views per format | PNG current | PNG combined | JPEG75 current | JPEG75 combined |
| --- | ---: | ---: | ---: | ---: |
| Sharp adjusted copies localized /12 | 2 | 4 | 2 | 3 |
| Feathered adjusted copies localized /12 | 0 | 0 | 0 | 0 |
| Sharp unadjusted copies localized /12 | 7 | 7 | 6 | 6 |
| Feathered unadjusted copies localized /12 | 4 | 4 | 3 | 3 |
| Ordinary views flagged /48 | 0 | 0 | 0 | 0 |

One PNG feathered positive received an additional pair at the wrong location. The sparse check proposed no pair on any ordinary view, including all 32 deliveries of the four repeated-structure photos. This panel shows that the upstream check handled those controls. It leaves the new comparison's false-alarm behavior on ordinary images that produce plausible sparse proposals untested. The copies are authored conventional edits and carry no information about AI authorship.

The CPU run measured all 192 deliveries in 29.5 seconds. The standard-library reviewer checked all rows and 102 pair-side overlap records using exact integer fractions. An exact local replay reproduced the entire per-view journal byte for byte.

## Ordinary proposal stress test

A separate proposal-only screen used the retained source cache. It selected the smallest encoded source digest per independent cache component and retained 151 of 155 groups after applying the serving size limits. The 302 ordinary deliveries used the decoded photo and a matched JPEG75 encode. The screen measured all views and recorded 33 RGB and 52 JPEG75 views with crowded repetition. Only three deliveries, from two source groups, had distinct sparse proposals. All three had an incumbent dense mask, so zero ordinary proposals entered the new fallback. The observed ordinary flag counts were two RGB and one JPEG75 in both arms. These results measure how often ordinary images reach the branch in this sample. They leave its error rate on eligible ordinary pairs unmeasured. The source cache qualifies licensed reference bytes and credited photographer separation. It does not establish camera provenance for every image. [Source qualification](image-fresh-source-cache-v39-results.md) · [Independent replay counts](evaluation-panels/image-copy-tone-transfer-2026-11-11.json)

## Reproduce

From the repository root, use the four bound Wikimedia JPEGs and eight retained qualified originals. The public receipt records the four acquired source URLs, photographer credits and image byte hashes. Save each acquired file and its source URL as `<slot>.jpg` and `<slot>.url` in `$sources`. The receipt's `public_photo_roster` supplies `title`, `author` and `license` for `$sources/roster.json`, in slot order. The retained trial is the `trial` directory under `image-bg-real-small-insertions-2026-11-11`.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_copy_tone_transfer freeze "$sources" "$archive" "$run"
backend/.venv/bin/python -B -m scripts.image_copy_tone_transfer run "$sources" "$archive" "$run"
backend/.venv/bin/python -B -m scripts.review_image_copy_tone_transfer "$run"
```

For the separate stress test, provide the locally retained `image-fresh-source-cache-v39/private-groups.json` as `$cache` and an unused `$stress` directory. The comparison reopens only the inputs shortlisted by the sealed proposal screen.

```sh
for stage in freeze screen compare; do
  backend/.venv/bin/python -B -m scripts.image_copy_tone_ordinary_stress "$stage" "$cache" "$stress" || exit
done
backend/.venv/bin/python -B -m scripts.review_image_copy_tone_transfer "$run" --ordinary-stress "$stress"
```
