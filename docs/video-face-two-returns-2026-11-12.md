# Two disjoint brief face changes in one video

The Full worker can already display a tracked face range and one supplemental brief face-change range. This comparison tests whether the same native measurements can support one more useful location when two short face swaps occur in a clip. It preserves the earlier locations and the analysis confidence. [Current six-picture evidence](video-face-six-picture-transfer-2026-11-12.md) · [Prior one-range ranking comparison](video-face-return-eligibility-2026-11-12.md)

## Frozen comparison

Each held video is an upright 640 × 360 H.264 4:2:0 delivery with 120 pictures at exactly 25 fps. Eight host groups were selected by a fixed metadata order from the existing confirmation archive. Their recorded original links and known donor keys are distinct. All sources had earlier project exposure. The edit combinations and placements were new to the project. The two source intervals are **2.080–2.320 s** and **3.120–3.360 s**. Each known six-picture swap uses a different publisher-rendered family and a corresponding genuine host recording. The preparation confirms that every inserted native picture differs from the corresponding original picture. [Source and rights](video-common-source-preparation-v89.md) · [Native input verification](video-native-inserts-v95-method.md)

Each group supplies eight delivered variants. These include unchanged and tandem-recompressed originals, one isolated swap at each placement, two swaps, a swap with a genuine same-source jump, a swap with ordinary illumination, and two genuine whole-picture cuts to another recording. The controls with a single swap test whether a nearby ordinary change acquires an additional false location. The held comparison uses the app’s existing commercially admitted face assets and the real Full probe, decode, shot, track, face and detect stages in an isolated process. [Appearance asset and usage terms](video-appearance-plan-v107-method.md)

The candidate uses the existing 12.5 Hz grid only after an earlier face range exists and the native short check has usable context. It reuses an already measured dense grid. If the sparse pass published a brief range, the dense grid can run within the original 12-second inspection budget. The additional range must clear the existing appearance-separation and duration rules, be separated from earlier intervals by at least two native pictures, and have a background lower-quartile correlation of at least 0.60 at both boundaries. It selects one additional range and carries zero AI-confidence weight.

A useful location overlaps a known swap by at least half temporal intersection over union. A gate frozen before scoring requires four useful additions across three sources and allows one additional ordinary range, one incorrect addition on a single-swap delivery, one incorrect addition on a dual-swap delivery, and zero additional genuine-cut ranges. Previously displayed results stay in the comparison.

## Held results

| Delivered condition | Clips | Authored swap ranges | Useful ranges before | Useful ranges with the second check | Additional wrong ranges |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged original | 8 | 0 | 0 | 0 | 0 |
| Recompressed original | 8 | 0 | 0 | 0 | 0 |
| Single early swap | 8 | 8 | 8 | 8 | 0 |
| Single late swap | 8 | 8 | 8 | 8 | 0 |
| Two swaps | 8 | 16 | 8 | 16 | 0 |
| Swap and same-source jump | 8 | 8 | 8 | 8 | 0 |
| Ordinary illumination and swap | 8 | 8 | 8 | 8 | 0 |
| Two genuine picture cuts | 8 | 0 | 0 | 0 | 0 |

The second check found all eight previously missed swap ranges in the eight dual-swap clips. All eight clips then showed both authored ranges. The frozen gate passed, with no additional ranges on the ordinary, single-swap, or genuine-cut cases. The earlier check already flagged one recompressed original. The 64 deliveries reuse eight host groups. The observed face changes carry zero weight in AI authorship confidence.

Independent readback recovered the same counts from the frozen observations and checked 7,680 exact native pictures and 24 source crops. It also reproduced the encoded bytes of all eight dual-swap and eight genuine-cut deliveries. One added range ran from 2.080 to 2.320 s on the exact source clock. Its before, during and after pictures came from 2.000, 2.160 and 2.320 s. [Frozen aggregate and independent replay](evaluation-panels/video-face-two-returns-2026-11-12.json)

## Local app verification

Enable `VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW=true` with the brief face-change review and appearance mode. The isolated Full worker ran the normal face detection, tracking and appearance paths with simulated prepaid checkout. On the held dual-swap clip, enabling the second check preserved the previous 3.120–3.360 s range and the whole-file decision, and added the 2.080–2.320 s range. The single-swap control kept one range. The genuine two-cut control showed no brief face-change range.

The serving adapter separately replayed all eight dual-swap files. Both interval endpoints, all three native anchors of each added range and both of its guarded background measurements matched the frozen comparison in 8/8 files.

The API, JSON export, standard PDF and signed package agreed on the displayed ranges and 18 original and cropped frame images across the three cases. Signed original replay recomputed the guarded background boundaries and verified the source carrier for the added range. The package committed to the two measurements separately. All three packages also verified in a separate process with networking and application runtime imports disabled. [Local app replay receipt](evaluation-panels/video-face-two-returns-app-2026-11-12.json)

## Reproduce locally

Use the retained licensed archive and a new private directory. The original frozen run used the app sources at `e4a6a5b` with the study runner and reviewer linked below. Use those app sources with the two current study scripts to reproduce the original freeze. The replay checks the source clocks and exact original pictures and crops. Per-source rows, images and native media remain local. [Study runner](../backend/scripts/video_face_two_returns.py) · [Independent reviewer](../backend/scripts/review_video_face_two_returns.py)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare score report; do
  backend/.venv/bin/python -B -m scripts.video_face_two_returns "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_two_returns "$run"
```
