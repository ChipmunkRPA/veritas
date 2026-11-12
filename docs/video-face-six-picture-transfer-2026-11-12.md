# Short face appearance changes across six encoded pictures

Both 12.5 Hz checks localized all 16 held six-picture swaps. The existing coarse and 8 Hz checks localized none. Both candidates added zero flags on 20 ordinary deliveries and four whole-picture cuts. The denser version of the existing rank calculation and the stricter comparison selected the same intervals. [Held results and exact replay](evaluation-panels/video-face-six-picture-transfer-2026-11-12.json)

## Frozen comparison

The earlier short-return check found nine-picture swaps in fixed 360p H.264 deliveries. This trial uses 0.240-second edits on a separate set of source recordings. Recent work on partially manipulated video tests consecutive-frame comparisons to locate temporal boundaries. Here the same admitted face and appearance weights supply every arm. A scene check constrains both boundaries using background structure outside the face. [Earlier H.264 result](video-short-return-encoded-360p-2026-11-11.md) · [Temporal localization research](https://arxiv.org/abs/2606.07090) · [Existing weight review](video-appearance-plan-v107-method.md)

The retained publisher media supplied two development source groups and four held groups. The held sources include two native 720p and two 1080p recordings at exactly 25 fps. All six have the publisher's head-rotation tag. Source links and known donor keys are disjoint from the five preceding short-return, cut-guard, resolution, H.264, and codec rosters. The known keys cannot establish separation by person or from model pretraining. Some footage had earlier exposure in other project work. The development clips run first to check the procedure. Neither their results nor the held observations set a new point. [Native source construction](video-native-inserts-v95-method.md) · [Publisher grants and known linkage limits](video-common-source-preparation-v89.md)

Each group supplied two publisher swap families at each of two exact intervals, **2.160–2.400 seconds** and **2.680–2.920 seconds**. All six selected native pictures differed in each authored insert. The placements use different phases of the 12.5 Hz scan. Each group also supplied its original, CRF 36 recompression of the CRF 23 base, and matched lighting, blur, and timing changes. Held pairs supplied four cuts between whole pictures of original recordings. The silent deliveries use 120 exact pictures at 640 × 360, H.264 4:2:0, and the same 25 fps clock. The original audio is outside the study.

The first comparison runs the existing coarse and 8 Hz appearance calculations. The density arm runs the same guarded rank scan at 12.5 Hz. The consecutive support arm uses the same 12.5 Hz pictures, requires all pictures in both flanks and the middle, and takes the smallest of three or four individual middle margins. This requires every measured center to support a range of 0.240 or 0.320 seconds. Successive observed pictures are 80 ms apart. The range uses their exact native timestamps. Both supplemental arms preserve the existing ranges, use the same strict 0.10 point, and suppress a candidate if it overlaps an existing range at 0.50 IoU or more. This controlled test uses directly observed single faces and the serving appearance calculation. Full application tracking and payment processing are outside the test scope.

The frozen gate requires at least six added correct ranges across at least three held source groups, at most two added ordinary ranges, no added hard-cut ranges, and 90% assessed coverage separately for swaps and ordinary controls. Each correct range needs at least 0.50 IoU with the known interval.

The first scorer stopped before saving a score when its hash function received a noncontiguous crop array. The continuation converted the same native pixels to bytes before hashing. It checked the one-line change and reused all 60 encoded files byte for byte. The original freeze and empty failed journal remain available locally.

## Held results

| Outcome | Current coarse and 8 Hz | 12.5 Hz rank | 12.5 Hz consecutive support |
| --- | ---: | ---: | ---: |
| Correct six-picture locations | 0/16 | 16/16 additional | 16/16 additional |
| Wrong intervals on swap views | 3/16 | 0/16 additional | 0/16 additional |
| Flags on ordinary views | 0/20 | 0/20 additional | 0/20 additional |
| Flags on whole-picture cuts | 1/4 | 0/4 additional | 0/4 additional |
| Assessed views | 40/40 | 40/40 | 40/40 |

The current 8 Hz check selected no range in these 40 views. Each denser arm found all four swap views in each of the four held groups. Both returned identical endpoints on all swaps. Eight ranges matched the authored start and end exactly. The other eight began and ended one native picture, 40 ms, later than the authored interval. Median IoU was 6/7, with a minimum of 5/7. The original coarse ranges, including one genuine cut flag, were retained. Each arm cleared the frozen gate.

One exact range is **2.160–2.400 s**, with the source pictures at **2.080**, **2.240**, and **2.400 s**. The denser rank pass returned positive margins from 0.339 to 0.679. The largest measured control margin was 0.010. The 107.2-second held pass scored 3,960 sampled picture positions on one CPU thread. It uses fixed 4.8-second H.264 delivery, one visible face, and a small number of source groups. It measures face appearance for review.

The independent replay matched all 4,800 native decoded pictures from the 40 held files and all 96 selected face crops. It reopened 576 qualified original or swap-source pictures and independently recreated the 16 swap and four whole-picture cut MP4 files byte for byte. This verified all 96 swapped native picture inputs at their authored intervals before encoding. Per-source media and evidence images stay local.

## Reproduce

Use the retained local archive and a fresh private output directory. The preparation verifies input picture hashes before composing each delivery. The reviewer opens the actual encoded input, checks the selected source crops, and independently reconstructs the known interventions.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare development held report; do
  backend/.venv/bin/python -B -m scripts.video_face_six_picture_transfer "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_six_picture_transfer "$run"
```
