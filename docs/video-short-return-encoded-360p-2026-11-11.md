# Short face-change localization on encoded 360p video

The preceding paired study found all 12 short swaps on downsampled 360p RGB pictures. This test qualifies the transfer on actual H.264 MP4 files through the local application's production face sampler, tracker and appearance review. [Prior resolution study](video-short-return-resolution-transfer-2026-11-11.md)

## Frozen panel

Four archived source groups, two native 720p25 and two native 1080p25, provide 32 six-second 360p25 files. The roster excludes the 18 sources in the preceding short-return, cut-guard and resolution trials, including overlaps in known source links or donor keys. Within each native size, the smallest eligible sources by archived byte size are selected before encoding or scoring. The source groups may have appeared in other project work. The labels describe the generated inserts and ordinary controls, and the files have no soundtrack. [Native source construction](video-native-inserts-v95-method.md) · [Source terms](video-common-source-preparation-v89.md) · [Weight review](video-appearance-plan-v107-method.md)

Each group supplies two swap families with nine-frame edits at fixed placements, from 2.680 to 3.040 seconds and from 3.000 to 3.360 seconds on the exact 25 Hz clock. Each group also supplies an unchanged recording, a strong recompression, and matched ordinary light, blur and time-step edits. Two source pairs supply two directions of genuine hard cuts each. All primary inputs use libx264, YUV 4:2:0, CRF 23 and fixed 25-frame keyframe spacing. The recompressed control re-encodes the first delivery at CRF 36. Source pictures and their clock are verified before the resize. A private preflight binds every encoded input hash, frame PTS and decoded RGB hash.

The check uses the same 0.10 response, half-open duration under 0.50 seconds, and duplicate suppression at IoU 0.50. The application's incumbent and the supplemental range are measured separately. Localization requires IoU of at least 0.50 against the labeled edit. The gate requires four additional correctly localized swaps across three sources, at least 80% assessed coverage in swaps and ordinary controls separately, at most one additional ordinary false range and zero additional hard-cut false ranges.

## Results

| Frozen outcome | Deployed incumbent | With supplemental range |
| --- | ---: | ---: |
| Short swaps correctly localized | 1/8 | 8/8 |
| Ordinary controls with false ranges | 2/20 | 2/20 |
| Hard cuts with false ranges | 0/4 | 0/4 |

The check passed the frozen gate. It supplied seven additional correct ranges across all four groups and no misplaced supplemental ranges. The app assessed all 32 files. The median added range had 0.636 IoU and the largest endpoint error was 80 milliseconds. The independent auditor matched all 4,800 decoded picture hashes and 42 source-frame and crop PNGs to the encoded uploads. The study covers six-second, fixed-frame-rate H.264 conversions of matched single-face footage. Native low-resolution camera footage, other encoders and crowded scenes remain outside this panel. [Aggregate and source-pixel audit](evaluation-panels/video-short-return-encoded-360p-2026-11-11.json)

## Local serving

The optional short-range check now accepts upright 640×360 at exact 25 fps with verified timestamps and enough single-face measurements. It uses the existing 2–8 second scan bound. Set `VERITAS_VIDEO_FACE_REVIEW_MODE=appearance` and `VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW=true` to activate it.

A second worker replay used the serving profile with no experimental override. One short swap, one unchanged original and one hard cut each ran with the option off and on. The decision and incumbent range agreed in every pair. The extra check marked **2.760–3.120 seconds** for the **2.680–3.040 second** swap and added no range to either control. Checkout and the unpaid webhook queued zero jobs, and each simulated paid event queued one job despite repeat delivery. No live payment was used.

The evidence API and JSON export agreed. The before, during and after frames and their three crops matched the encoded input pixel for pixel and appeared in the PDF. A signed ZIP passed offline verification against the original upload. The 360p packet uses the existing version 12 layout and this release's verifier.

## Reproduce

Use the retained archive and a fresh private output directory. The scoring worker uses local fake Stripe responses and the actual billing and analysis entry points. It limits Full-profile activation to the face detection, tracking and appearance paths. The independent reviewer reopens the encoded files, checks the selected source timestamps and face crops and recomputes outcomes from the worker's incumbent and added ranges. Fresh runs bind their own source revisions.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_short_return_encoded_transfer freeze "$run_dir" --archive "$archive" --preceding "$paired_resolution_freeze"
backend/.venv/bin/python -B -m scripts.video_short_return_encoded_transfer prepare "$run_dir" --archive "$archive"
backend/.venv/bin/python -B -m scripts.video_short_return_encoded_app "$run_dir"
backend/.venv/bin/python -B -m scripts.review_video_short_return_encoded "$run_dir"
```
