# Background support for short face-change ranges in VP9 and HEVC

The earlier transfer found all eight six-picture swaps in VP9/WebM and HEVC/MP4. The dense fallback also flagged two genuine cuts in each codec. The same two cut directions triggered on matched H.264 copies. A subsequent H.264 trial kept all ten useful dense ranges when both boundaries met a stricter background requirement. That trial had no false dense ranges to correct. This comparison applies the same fixed requirement to the two other codecs. [Earlier codec diagnostic](video-face-six-codec-transfer-2026-11-12.md) · [Frozen H.264 background comparison](video-face-scene-continuity-2026-11-12.md)

## Fixed check and held inputs

Keep the first face appearance comparison and the 8 Hz scan. When the original 8 Hz scan assessed a candidate and supplied no distinct range, review the selected range from every second native picture. Keep that range only when the lower quartile of textured background tile correlations reaches .60 at both measured boundaries. Exclude tiles that intersect the observed face, with the existing margin. The guard runs on the original selected range and cannot choose a different winner. The review is a local source observation with zero AI confidence weight.

The source roster excludes 77 previously used short-localization and background-challenge groups by known original links and donor keys. The retained archive has ten eligible groups after these exclusions. Select three disjoint genuine source pairs using the earlier fixed palette and outer-grayscale distance, without face detector outputs. All publisher inserts use their measured native frames. Each output time selects the nearest authenticated source frame, with an earlier-frame tie break and no interpolation. The resulting 84 files each have 120 native upright 640 × 360 pictures at exact 25 fps with 4:2:0 chroma.

Each codec has 12 known publisher swaps over **2.160–2.400 s** or **2.680–2.920 s**, 12 genuine whole-picture cuts at the same intervals in the paired recordings, and 18 unchanged, tandem-compressed and ordinary illumination controls. The held groups are separate from this study's exposed discovery examples by documented source and donor keys. Earlier unrelated project use and unknown identity links remain possible. [Media rights and scope](video-common-source-preparation-v89.md)

The gate requires at least four new correctly localized swaps over three sources, up to one useful swap lost versus the unrestricted dense pass, no added cut flags, up to one extra other control flag, and at least 90% measured fallback coverage on swaps and other ordinary inputs. A useful location overlaps at least half its temporal union with the inserted range. The two exposed discovery cuts must also lose their additional dense ranges. The short comparison uses directly observed faces and the existing appearance calculation. The coarse benchmark comparator omits full application tracking.

## Held results

| Held outcome | VP9/WebM | HEVC/MP4 |
| --- | ---: | ---: |
| Useful swap ranges before and after | 0/12 to 10/12 | 0/12 to 10/12 |
| Sources with useful new ranges | 5/6 | 5/6 |
| Useful dense swap ranges withheld by the guard | 0/10 | 0/10 |
| Additional dense ranges on genuine cuts | 0/12 | 0/12 |
| Additional dense ranges on other controls | 0/18 | 0/18 |
| Fallback assessed on swaps and other controls | 30/30 | 29/30 |

Both codecs met all seven frozen requirements. Five selected ranges per codec matched the authored bounds exactly. Five had both endpoints one native picture later. Median temporal overlap was 6/7 and the minimum was 5/7. The earlier checks kept one VP9 cut flag, two HEVC cut flags and three HEVC flags on other ordinary deliveries. On the exposed discovery set, both codecs retained all eight previously localized swaps and withheld the two added dense cut ranges. The held set contained no additional false dense ranges for the guard to remove.

The independent reconstruction rebuilt all 84 fresh files byte for byte from 792 native archived picture records and independently checked the 60 authored intervals. The first reconstruction attempt stopped on a byte mismatch before completing its roster. The complete repeat used the same source and encoding recipe. The separate decision verifier checked 10,080 delivered pictures, 60 exact crops, 160 source-bound background pictures from the held and exposed sets, and 444 dispositions. The production face measurement separately reproduced one new swap at **2.160–2.400 s** in each codec, one unflagged original in each and one withheld exposed cut range in each. These six measurements ran in 2.99 to 4.81 seconds under the existing 12-second budget. [Public decision ledger](evaluation-panels/video-face-codec-background-transfer-2026-11-12.json)

## Local app scope

With short face review enabled, the Full and Deep worker can use the background guard on the tested WebM VP9 or ISO MP4 HEVC profile. It preserves the existing H.264 paths and runs within the original time allowance. The normal evidence view and report show the matched timestamps and before, during and after face crops. Signed evidence replay verifies the original codec and carrier, the exact source pictures and crops, and both background measurements.

In the isolated Full worker, two held swaps reached **2.160–2.400 s**. Neither held original nor either fresh ordinary cut received an added range. The guard withheld both exposed genuine cut ranges. These runs used simulated checkout.

Both positive runs showed the same interval and source crops in Chrome, the PDF,
and the JSON. The two signed exports replayed against the original uploads,
including the background measurements.

## Reproduce

Use the existing native archive, pinned local weights and retained previous rosters. The independent authoring command rebuilds every delivery and checks the authored RGB edits and file bytes. The decision replay checks source timestamps, actual video carriers, exact original crop pixels and the background quartiles.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare score report; do
  backend/.venv/bin/python -B -m scripts.video_face_codec_background_transfer "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_codec_background_transfer authoring "$run"
backend/.venv/bin/python -B -m scripts.review_video_face_codec_background_transfer decisions "$run"
```

The public ledger can be replayed with the Python standard library alone. It contains anonymous study ordinals, measured frame ranges and aggregate results.

```sh
python3 -B -I -S backend/scripts/review_video_face_codec_background_ledger.py \
  docs/evaluation-panels/video-face-codec-background-transfer-2026-11-12.json
```
