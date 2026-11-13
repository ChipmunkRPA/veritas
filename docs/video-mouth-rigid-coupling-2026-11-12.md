# Rigid head registration for short mouth redraws

Neither experimental check located a redraw in the ten held videos that retained a verified mouth-release event. The jaw-motion check flagged one untouched recording and two ordinary retimes. The angular-head-motion check flagged two untouched recordings. Both stay outside the app. [Frozen arithmetic and native exhibit audit](evaluation-panels/video-mouth-rigid-coupling-2026-11-12.json)

## Test design

Recent work argues that local synthesis can disrupt how lip movement relates to head motion. This experiment independently tested whether a smaller cue survives on short, synchronized, same-speaker redraws. It used the existing rights-reviewed Face Landmarker and none of the published research datasets, classifier code, or weights. The prior local optical-flow model predicts lip deformation from surrounding motion. This test instead fits a proper rigid transformation to 12 upper-face landmarks between consecutive native pictures. It measures mouth and jaw movement after that transformation, and separately extracts angular head motion from the fit. [Head and lip research](https://proceedings.mlr.press/v306/she26a.html) · [Prior optical-flow experiment](video-mouth-jaw-transfer-2026-11-11.md)

The candidate scans 12-, 16- and 20-picture intervals with nine adjacent edges on each side. One arm compares signed opening velocity with chin and jaw velocity. The other compares opening speed with angular head speed. Both require movement support in all three windows and positive correlation on both flanks before proposing a localized loss of agreement. Each arm keeps at most one range. The submission alone provides its pictures and landmark observations. Audio, authorship labels and edit boundaries enter only evaluation. A result needs at least 0.50 temporal intersection over union with the actual half-open edited interval.

Four GRID ordinary views from two speakers and six Lombard ordinary views from a different speaker set the internal points before held scoring. The frozen points are **0.9603** for jaw motion and **0.8379** for angular head motion. The 92 held files consist of 11 source-time neural redraws, 33 matched ordinary controls, and 48 genuine recordings from 12 additional Lombard speaker groups. Ten redraws from six held speaker groups had independently observed source-time mouth releases and unchanged audio. The new genuine groups had not been used to set this detector’s points. The 11 positive deliveries had earlier project exposure and use one renderer. [Native neural test](lombard-neural-mouth-redraw-2026-11-12.md) · [Additional source roster](short-lip-release-timing-2026-11-12.md) · [Lombard publisher and licence](https://spandh.dcs.shef.ac.uk/avlombard/)

## Held results

| Outcome | Jaw-motion agreement | Angular-head-motion agreement |
| --- | ---: | ---: |
| Measured | 33/92 | 31/92 |
| Verified source-aligned redraws measured | 3/10 | 3/10 |
| Verified source-aligned redraws localized | 0/10 | 0/10 |
| All neural redraws localized | 0/11 | 0/11 |
| Untouched recordings flagged | 1/59 | 2/59 |
| Ordinary resamples flagged | 0/11 | 0/11 |
| Ordinary delayed-mouth controls flagged | 2/11 | 0/11 |

The independent arithmetic replay checked all 204 arm decisions. It reopened 15 exact source crops across all five flagged ordinary ranges. The held roster had 5,697 picture positions, including 89 files with the supported picture clock and 2,237 accepted geometry observations. Three untouched files contained fewer than 48 pictures. An audit of two otherwise usable tightly framed Lombard originals found the selected forehead landmark outside the image on every picture. Both still had one measured face. This prompted a separately frozen scope probe below.

The 60 licensed CREMA-D transfer views from ten additional source groups are outside the frozen clock scope. Their archived first picture steps are 33 ms, below the 40 ms minimum. They contribute zero measured outcomes and do not enter the above denominator. This frozen test cannot establish performance at 30 fps or on longer, naturally moving speakers. [CREMA-D terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

## Interior-landmark coverage probe

An independent face read found a single tracked face in each of 129 pictures across the two tightly framed examples. The forehead point lay beyond the frame in all 129. A separate post-result freeze removed only that point and retained the original points, clocks and comparison rules. It remeasured the same 92 held files. Accepted landmark observations increased from 2,237 to 5,551. Jaw-motion coverage rose from 33 to 89 files and head-motion coverage from 31 to 86. Both arms measured all ten source-aligned redraws and localized zero. Jaw-motion flags increased from three to six ordinary files. Head-motion flags increased from two to seven and included four untouched recordings. Head motion also produced one misplaced neural range. This exploratory repeat isolates a crop limitation. The substantially higher coverage did not improve localization on these sources.

## Local replay

Use the retained private Lombard redraw folder as `lombard`, the 48-member earlier-copy intake as `fresh`, and its private short timing measurement folder as `observations`. Use a new private `run` directory. The pinned macOS Face Landmarker uses its local graphics context even with CPU delegation. Original media and crops stay private.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.video_mouth_rigid_coupling "$phase" "$run" "$lombard" "$fresh" "$observations" || exit 1
done
backend/.venv/bin/python -B -m scripts.audit_video_mouth_rigid_coupling "$run"
for phase in freeze run; do
  backend/.venv/bin/python -B -m scripts.video_mouth_rigid_interior_probe "$phase" "$run" "$probe" || exit 1
done
python3 -B -S -m scripts.review_video_mouth_rigid_coupling "$receipt" --assemble "$run" --interior-probe "$probe"
python3 -B -S -m scripts.review_video_mouth_rigid_coupling docs/evaluation-panels/video-mouth-rigid-coupling-2026-11-12.json
```
