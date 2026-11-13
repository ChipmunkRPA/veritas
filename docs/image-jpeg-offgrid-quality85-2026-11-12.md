# Localizing larger copied areas in standard JPEG85 files

The app now extends the between-block JPEG copy check to a standard quality-85 compression profile. The source-separated confirmation raised useful locations from 4 to 11 of 24 controlled sharp copies and added zero flags on 45 ordinary deliveries. The check shows two exact source crops and compares how closely their colors agree inside each crop and differ around their edges. Either area may have come first. These findings contribute zero weight to AI authorship confidence. [Evaluation receipt](evaluation-panels/image-jpeg-offgrid-quality85-2026-11-12.json)

## Two frozen trials

The existing translation and exact encoded brightness-block comparisons run first. The new path runs when both complete with no pair. The first trial used the existing 32 px footprint and twenty licensed Wikimedia photos outside the eight-source public replay. The photos had previous project exposure. It added two useful 40 × 48 locations and five useful 64 × 64 locations, bringing the total from 10 to 17 of 24 authored edits. There was one incumbent flag and zero added flags across 36 ordinary deliveries. The trial required three added locations at each size and missed that requirement for small edits. The first launch stopped before scoring because the runner's table digest had one missing character. Correcting the digest produced a new freeze with the same sources, edits and gate.

The follow-up froze a 56 px minimum on each axis of the selected crops. It kept the existing search ordering and withheld smaller pairs. Its 12 edit sources come from the archived, rights-verified photo cache and its nine structural controls come from the Wikimedia roster. The 21 credited contributors and their encoded source hashes are distinct from the first trial's. Some sources may have been used in other project studies. Each edited source supplied 64 × 64 and 72 × 80 sharp copies plus an unchanged view, a contrast adjustment and local blur. Each structural source supplied a re-encoded view. The two rectangles needed at least 0.50 intersection-over-union with the authored regions and the target crop had to recover at least half of the actual changed pixels. The fixed gate required useful gains on four or more source groups, at least two at each size, at most one additional ordinary source group, zero additional structural source groups and at most one incorrectly localized edited view. [Archived photo cache rights](image-fresh-source-cache-v39-results.md) · [Wikimedia source credits and exact bytes](evaluation-panels/image-affine-native-scale-2026-11-11.json)

| Confirmation outcome | Existing check | With the JPEG85 check |
| --- | ---: | ---: |
| Useful 64 × 64 copies | 1/12 | 4/12 |
| Useful 72 × 80 copies | 3/12 | 7/12 |
| Flagged unchanged, contrast and local-blur views | 0/36 | 0/36 |
| Flagged repeated-structure views | 0/9 | 0/9 |

Seven new useful locations came from five source groups. No added edited view had an incorrect location. All 45 ordinary views reached the new comparison. Four edited views had already been localized before the fallback. The native verifier checked 22 crop assets and rejected moved crops and altered color measurements. The exact serving replay reproduced all 69 frozen result records. The measured cases cover sharp intra-image copying under controlled JPEG encoding. They provide no measurement of generator fills, independent camera origin or customer-upload error rates.

## Serving and readback

The existing `VERITAS_IMAGE_JPEG_OFFGRID_COPY_REVIEW=true` enables the added profile in the development environment. The quality-85 scope requires the exact measured brightness and color quantization tables, upright RGB 4:2:0 JPEG, a short side of at least 384 px and a long side of 513–1,024 px. The earlier JPEG75 and JPEG90 profiles retain their measured 32 px minimum. Source verification enforces the quality-85 crop footprint against the submitted JPEG bytes.

An isolated Full worker accepted a positive and an ordinary control after simulated paid checkout. Both unpaid events left the analysis queue empty. This CPU smoke disabled the other model lanes. The positive produced two 64 × 58 crops with the same source pixels in the API and PDF. The API evidence, JSON export and signed evidence archive agreed, and native source replay recalculated the inside and surrounding color differences. The control recorded a completed check with zero crop assets. The isolated API, JSON and PDF were checked for model-name redaction.

## Reproduce

Use the recorded dependency versions. In a scratch checkout of the pre-extension revision `9810bcabc712fa39164caaef247534f93778cfd0`, copy in the two evaluation runners and use the licensed archive or the exact public files. The first runner's `acquire` mode can retrieve its twenty pinned Wikimedia deliveries. Place the confirmation's Wikimedia deliveries at the documented `source-NN.jpg` filenames and supply the bound v39 local cache for the 12 edited sources. The receipt records the verified hashes and attribution. Keep the acquired JPEGs and derivative pixels in scratch storage.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze run review; do
  python -B -m scripts.image_jpeg_offgrid_quality85 "$stage" "$public_sources" "$first" || exit
done
for stage in freeze run review; do
  python -B -m scripts.image_jpeg_offgrid_quality85_confirmation "$stage" \
    "$first/freeze.json" "$cache" "$public_sources" "$confirmation" || exit
done
```

The two `review` commands regenerate the held measurements and verify the original bytes, crop coordinates and color readings. Replay the historical freezes with the pinned pre-extension app files. The separate serving replay compares the shipped method on the narrower frozen panel.
