# Tracking mouth-region points through multiple video frames

This experiment tests whether points in a short edited mouth region become harder to follow over consecutive pictures than comparably textured points on nearby cheeks. It compares direct long-hop tracking with the endpoint reached by tracking each consecutive picture, and independently brings both routes back to their start. Related work has used forward–backward error to measure tracking failure and temporal facial feature drift for face-forgery detection. The test here measures geometric correspondence in a single uploaded file with existing rights-reviewed face geometry. It uses none of the research detector weights, no earlier copy, and no audio or marked location. [Forward–backward tracking research](https://dspace.cvut.cz/entities/publication/4f34fd8a-dffe-4857-96c2-42ed3d5a8449) · [Video-level blending and feature drift research](https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_Generalizing_Deepfake_Video_Detection_with_Plug-and-Play_Video-Level_Blending_and_Spatiotemporal_CVPR_2025_paper.pdf) · [Face geometry licence](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

## Frozen comparison

On an eye-registered picture plane, match mouth and cheek corners with similar initial strengths. Follow each set over three and four native picture intervals. The first arm measures loss of surviving tracks relative to the cheek set. The second measures spatial disagreement between the two routes and their round trips. Both scan short contiguous ranges, requiring measured flank and interior support in one shot and a positive change on both horizons. The strongest range is chosen before the edit label is joined. A useful range overlaps at least half of the actual edited interval by temporal intersection over union.

The third-largest ordinary response in 56 GRID development files sets each internal operating point, subject to fixed floors. The 146 held files include 11 earlier LivePortrait redraws and 33 matched originals/resamples/ordinary timing edits, 48 genuine Lombard originals, 12 codec and genuine-cut controls, 30 new MuseTalk deliveries, and 12 new ordinary temporal-processing controls. The latter controls use two interventions over the same 16-picture mouth-region feather on one source from each of six speakers outside the MuseTalk group: same-time subpixel resampling and an ordinary centered two-picture temporal blend. Both controls preserve all source timestamps and all pixels outside the local operation. [Earlier independent local speech-event audit](lombard-neural-mouth-redraw-2026-11-12.md) · [New MuseTalk panel and native audit](lombard-fresh-musetalk-2026-11-13.md) · [Lombard source and terms](https://zenodo.org/records/3736465) · [GRID source and terms](https://zenodo.org/records/3625687)

Ten earlier LivePortrait redraws and one new MuseTalk redraw met the frozen speech-event and control requirements. The new MuseTalk group is independent of both the GRID development and the earlier neural source groups. Two further new MuseTalk clips retained the local speech event but failed a matched control, so they are reported separately. Earlier-conditioned redraws are retained in the general neural denominator. Ordinary controls and variants from the same recording remain correlated. A local event check does not establish synchronization through the rest of a clip.

## Held results

| Check | Strict LivePortrait redraws located | Strict MuseTalk redraws located | Ordinary alerts | ROC AUC on measured strict redraws vs. ordinary |
| --- | ---: | ---: | ---: | ---: |
| Track survival loss | 0/10 | 0/1 | 1/123 | 0.683 |
| Disagreement between direct and chained routes | 1/10 | 0/1 | 8/123 | 0.566 |

Both arms measured 9/10 strict LivePortrait redraws, the strict MuseTalk redraw, all three MuseTalk clips that retained a local event, and 118/123 ordinary files. They measured 22/23 total neural deliveries. The second arm produced two correct ranges across all neural deliveries and one misplaced neural alert. Six of its eight ordinary alerts were on untouched original recordings. The other two came from one resample and one ordinary retime. Neither arm flagged any of the 12 newly prepared controls or the 12 held recompressions and genuine cuts.

The one strict LivePortrait hit covered **0.830–1.495 s** for an authored range of **0.789–1.453 s**, for temporal IoU 0.882. The strongest strict MuseTalk trajectory-loss range covered **0.374–0.706 s**, outside its authored **1.038–1.702 s** interval. The second arm's proposal covered **0.955–1.287 s**, for IoU 0.333. Both missed before applying the frozen points. The results do not support enabling either cue. They suggest a ranking signal for some edits in the survival arm, with weak interval agreement and no demonstrated second-generator transfer. [Complete outcome ledger](evaluation-panels/video-mouth-track-cycle-2026-11-13.json)

## Replay

The independent readback checked 47 exact native frame/crop commitments across 19 proposed ranges, and re-decoded all 670 pictures of the 12 new ordinary control files. It independently reconstructed the local operation and confirmed 192 changed pictures with exact unchanged flanks. Scoring committed 9,152 verified held picture timestamps. The isolated public reader reproduced the native readback's decision totals without third-party imports.

The native media, committed altered pixels and per-frame trajectories stay in the private runs. Use the retained prior Lombard redraws, fresh source bank, and independently audited MuseTalk study with a new private output directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare development held; do
  backend/.venv/bin/python -B -m scripts.video_mouth_track_cycle "$phase" "$run" "$lombard" "$fresh" "$muse" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_track_cycle "$receipt" --assemble "$run"
python3 -B -IS backend/scripts/review_video_mouth_track_cycle.py "$receipt"
```
