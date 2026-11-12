# Repeat appearance measurements on recompressed video

Four ordinary tandem recompressions triggered the current coarse tracked face check while their matching original deliveries triggered none. This experiment repeats the same face measurement on each selected window after reducing fine spatial detail. It also separates a single heavy encode from a heavy second encode and measures the effect on matched swaps. [Discovery outcomes](video-face-return-eligibility-2026-11-12.md)

## Frozen method

Build exact 120-picture, upright 640 × 360 H.264 4:2:0 deliveries at 25 fps. Rebuild a moderate direct encode and independently produce a stronger direct encode from the same native RGB sequence. Encode the moderate delivery a second time at each quality setting. The quality settings are CRF 23 and CRF 36. For the known swaps, replace the fixed native interval using the publisher's matching swap pictures, then repeat all four deliveries. The 23-quality outputs must reproduce the earlier archived MP4 hashes. The four exposed heavy tandem originals must reproduce the exact initial false-alert files. The recipe records which replacement pictures actually change. [Exact source timing and rights review](video-common-source-preparation-v89.md) · [Held source groups](video-native-background-challenge-2026-11-12.md)

The four discovery groups are separate from the twelve held groups by the archive's recorded links and donor keys. The held sources had earlier project exposure in the direct CRF 23 delivery. The comparison covers new delivery treatments and paired changes on those sources. Recent compression research similarly measures true H.264 encodes at multiple quality levels with matched controls. Its models and dataset are outside this experiment. [Aliyev and Rustamov, November 2026 revision](https://arxiv.org/abs/2608.28685v2)

Run the actual CPU Full video probe, decode, shot, track, face and appearance stages in isolation. When the current coarse stage selects a range, use its recorded native pictures, exact aligned crops and track membership. Keep all center triplets and before, during and after groups fixed. Repeat the same appearance features on a 56-pixel area downsample restored to 112 pixels with linear interpolation. Suppress that coarse range and its dependent timing refinement only if the repeated score falls at or below the existing internal point. Keep every existing short range. This arm tests appearance stability at a different effective resolution. Compression, motion, focus, or a small real edit can each cause instability. The arm has no authorship-confidence weight. [Source-bound coarse measurement](video-v17-sface-confirmation-evaluation.md) · [Commercial checkpoint qualification](commercial-model-audit.md)

Advancement requires removing at least two ordinary alerts from two held groups, assessing at least four useful coarse swap ranges, and preserving every useful coarse or complete swap result. A useful range must overlap the known interval by at least half temporal intersection over union.

## Held outcomes

| Treatment | Ordinary alerts, current → repeat | Useful swap locations, current → repeat |
| --- | ---: | ---: |
| Direct CRF 23 | 0/12 → 0/12 | 12/12 → 12/12 |
| Tandem CRF 23 | 0/12 → 0/12 | 12/12 → 12/12 |
| Direct CRF 36 | 6/12 → 2/12 | 12/12 → 12/12 |
| Tandem CRF 36 | 6/12 → 5/12 | 12/12 → 12/12 |
| All treatments | 12/48 → 7/48 | 48/48 → 48/48 |

The repeat removed five held alerts across four source groups, including one group that gained on both heavy deliveries. It preserved all 48 useful held swap locations. These are repeated deliveries of twelve source groups with swaps by two publisher methods. In the exposed discovery groups, direct CRF 36 also triggered all four originals. The repeat removed two of the four earlier tandem alerts and none of the four direct alerts there. The affected groups establish that heavy encoding can be sufficient to trigger the current review even without a second encode.

The candidate cleared its frozen criteria for this exact delivery profile. The study tested the current stages in isolation and modeled the supplemental display decision. [Aggregate and independent replay](evaluation-panels/video-face-compression-stability-2026-11-12.json)

The CPU screen completed 128 cases in 681 seconds. Independent authoring rebuilt all 128 files byte for byte from 1,920 verified original pictures and 280 publisher pictures. A separate process recreated 1,812 selected face-crop occurrences from 84 coarse ranges, reproduced both responses on the exact admitted weights, and rechecked the held decisions.

## App verification

Set `VERITAS_VIDEO_FACE_DETAIL_REPEAT_REVIEW=true` with the appearance review enabled to run the optional check. The app uses the existing coarse selection and its native source frames. It runs the repeat after the brief face check so the original coarse interval remains the basis for deciding whether a separate short range is distinct. When the optional repeat cannot finish, the measured review range remains available and the coverage identifies the unavailable repeat. The app, PDF, and JSON identify which original time interval the repeat checked and whether its appearance change persisted. The measurement carries zero overall confidence weight.

Twelve simulated paid worker analyses compared six files with the option off and on. Two genuine heavy encodes lost their earlier coarse alerts. Their two matched publisher swaps kept the exact intervals and timing refinements. A moderate direct genuine delivery remained clear. A separate brief swap kept its v15 interval and source-bound evidence. The four actual lower-detail responses and aligned crop hashes matched the held study. All six analysis decisions matched their paired results. Five reports and signed packages verified against the original uploaded videos, and the JSON evidence matched the API. Chrome verified the ordinary view of a withheld interval and a retained swap. [App verification receipt](evaluation-panels/video-face-compression-app-2026-11-12.json)

## Replay

Use the retained native archive, both prior panel directories, and the original comparison code at `47b3ff5ae4fd6e9ddf734ec90b68dc79bce85614`. The independent reader rebuilds the delivery bytes and recomputes the selected native crops, both appearance scores, and the range decisions. The public output contains aggregate counts and commitments.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze prepare score; do
  backend/.venv/bin/python -B -m scripts.video_face_compression_stability "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_compression_stability authoring "$run"
backend/.venv/bin/python -B -m scripts.review_video_face_compression_stability outcomes "$run" --publication "$receipt"
```
