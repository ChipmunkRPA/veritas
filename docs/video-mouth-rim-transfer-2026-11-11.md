# Spatial mouth-boundary review on short neural redraws

This comparison measures whether color or fine-texture steps around a moving mouth can locate a neural redraw when the soundtrack keeps its original timing. Prior motion reconstruction found zero of 32 neural redraws and proposed four ordinary ranges. This pilot uses measurements from single pictures and then checks whether they persist in a short time range. It uses no weights for the boundary measurement. Earlier boundary-focused research and work on seamless lip synthesis motivate testing the cue against ordinary local changes. [Boundary research](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Face_X-Ray_for_More_General_Face_Forgery_Detection_CVPR_2020_paper.html) · [Recent lip synthesis research](https://arxiv.org/abs/2608.18832) · [Prior motion comparison](video-mouth-reconstruction-transfer-2026-11-11.md)

## Source and fixed method

The trial reuses the frozen GRID source roster and the same eight source components as the prior comparison. Four components set the measurement points and four supply only held responses for this method. Each of 16 clips has 11 exact native versions. The set contains two neural renderer families at two requested timings, paired resampling and brightness controls, original speech and pictures, ordinary picture retiming, and an audio-only retime. Neural edit boundaries are 1.080–1.880 s. Picture clocks, exact RGB and archived changed-pixel masks establish the edited views. An earlier geometry audit describes which expression-driven redraws remain compatible with source-time mouth motion. It does not establish phoneme agreement. The panel was used for other project tests. The separate reserve was not scored. [GRID source and terms](https://zenodo.org/records/3625687) · [Native source construction](grid-native-clips-v149-result.md) · [Renderer terms and exclusions](video-timing-coverage-and-codec-holds-2026-11-11.md) · [Movement verification](grid-mouth-motion-v158-result.md)

Use every second native picture, one verified face, and the same 96-pixel eye registration and mouth location as the earlier direct comparison. The face detector is pinned to its commercially permitted checkpoint. At each of three fixed mouth-centered ellipses, sample four profiles with three-pixel radial spacing across eight angular sectors. Compare the middle CIELAB chroma change against both adjacent changes. A separate arm measures the middle change in log local luma-residual energy. It takes the absolute value after forming the signed sector median and subtracts the larger adjacent change. The joint arm averages both responses before temporal scoring. Each arm selects one range with the strongest middle median above both flank medians. All measurements in one range use the same ellipse. No counterpart, known mask, soundtrack, or edit boundary enters the score. [Face model license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

For each arm, select the third-largest measured ordinary development response and require strict exceedance. A useful range needs at least 0.50 native-time intersection over union with the known edit. The joint arm must add at least two useful ranges across two groups, including one source-time-compatible expression redraw, and add at most two ordinary ranges. The individual components report different error patterns. The study does not use them to select an alternative joint method after seeing held responses.

## Held outcomes

| Arm | Views measured | Correct redraw ranges /32 | Misplaced redraw ranges /32 | Ordinary flags /56 |
| --- | ---: | ---: | ---: | ---: |
| Direct native-picture comparison | 88 | 1 | 0 | 0 |
| Chroma boundary | 88 | 1 | 3 | 5 |
| Fine-texture boundary | 88 | 0 | 0 | 0 |
| Joint boundary | 88 | 0 | 0 | 0 |

The chroma check found a different redraw from the direct comparator. It localized an earlier-driven speech-conditioned redraw at **1.120–1.840 s**, which overlaps 90% of the authored **1.080–1.880 s** interval. The two checks taken together would find two redraws and flag five ordinary views. The ordinary flags came from one original, its visually identical audio-only variant, two resampling controls and one picture retime. All four arms missed the seven held source-time-compatible expression redraws. The joint arm did not meet the advancement criterion. The additional boundary check stays experimental. [Exact aggregate and verification receipt](evaluation-panels/video-mouth-rim-transfer-2026-11-11.json)

Both phases checked 6,688 decoded native picture inputs. The held phase ran on CPU in 58.34 seconds with 117,063,680 bytes of reported maximum RSS. The audit matched every one of the 176 direct measurements to the earlier run, verified all 88 held file hashes, and re-decoded 30 evidence frames at their exact source times. It checked two native edit masks. All 2,796 changed pixels in the selected frame of the correct chroma example lie inside the recorded inspection crop. That crop covers the measured mouth area and does not estimate a pixel-level edit mask. All 16 original/audio-only pairs returned identical scores. The data contain two renderer families and short recorded GRID prompts. Transfer to fresh sources, ordinary camera cuts and more recent rendering methods remains unmeasured.

An optional shell resource wrapper returned an error after writing the complete development journal. Its system accounting call was unavailable. Calibration consumed the verified completed journal, and the held phase used a Python timing wrapper.

## Replay

Use the retained archive and the previous direct comparison directory. The new freeze checks the prior methods and model bytes and binds the exact prior source partition. The separate replay redoes the outcome arithmetic and checks file hashes, native clocks and selected evidence pixels. Media, source identifiers and native crops stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_mouth_rim_transfer freeze "$run" --prior "$prior"
for stage in development calibrate held readout; do
  backend/.venv/bin/python -B -m scripts.video_mouth_rim_transfer "$stage" "$run" || exit
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_rim_transfer "$run" --prior "$prior"
```
