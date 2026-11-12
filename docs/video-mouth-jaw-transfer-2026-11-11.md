# Mouth and jaw motion on short retalking edits

The visual motion cue found one additional retalking range on the second source bank, with substantially more false ranges on ordinary footage. Both checks missed all seven independently qualified original-time redraws on the first bank. The new cue stays in research. [Aggregate and source-pixel replay](evaluation-panels/video-mouth-jaw-transfer-2026-11-11.json)

## Frozen comparison

LipDA's authors motivate comparing lip motion with the surrounding face and head. Their method uses separate deep lip and 3D pose encoders and a supervised contrastive objective. This experiment independently tests a CPU approximation. It reuses the commercially admitted face detector to align the eye positions and measures signed optical flow separately inside the lips, around the chin and cheeks, and at a stable upper-face reference. A ridge regression estimates the three measured lip deformations from ten jaw and cheek motion components. It was fitted on 288 observations from eight genuine GRID development recordings. The candidate searches for a rise or fall in the prediction residual relative to both flanks of a 0.40–0.88-second range. The direct comparator measures differences between neighboring aligned mouth pictures. Both arms use the same observed picture clock, alignment, support rules and sampled shot boundaries. [LipDA primary paper](https://proceedings.mlr.press/v306/she26a.html) · [Admitted face weight terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

The eight held GRID recordings are separate from fitting by the frozen connected source components. They supply 32 verified neural redraws and 56 matched ordinary views. Seven original-time expression-driven redraws independently match the visible mouth trajectory and preserve the original sound. That qualification covers visible timing. Eight actual H.264 recompressions retain the original soundtrack. Four visual-only hard-cut controls replace an exact 20-picture range with original pictures from another held source component. The cut controls use a verified lossless RGB encode. The whole GRID panel has previous project exposure, including the earlier direct, reconstruction and rim checks. [GRID source and terms](https://zenodo.org/records/3625687) · [Exact native construction](grid-native-clips-v149-result.md) · [Visible-motion qualification](grid-mouth-motion-v158-result.md)

The same regression and operating points transfer to 60 native views from ten CREMA-D confirmation groups across three previously identified connected contexts. That second bank is separate from the GRID fitting corpus and was previously used to test a different mouth/audio model. Its 20 neural edits are divided between original-time and 320 ms earlier conditioning. The archived audit verified the pixel masks, the actual source-clock boundaries and the unchanged PCM. The verified source-time conditioning records the requested timing. The separate direct and jaw arms set their own internal point from the third-largest ordinary GRID development response, then apply it unchanged. Correct localization needs at least 0.50 IoU against the exact half-open edit. [CREMA-D source and terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Native construction and PCM audit](native-mouth-clips-v120-complete.md) · [Earlier audiovisual comparison and timing coverage](native-mouth-calibration-and-transfer-v129-results.md)

## Results

| Held outcome | Direct picture comparison | Mouth and jaw motion |
| --- | ---: | ---: |
| GRID redraws correctly localized | 1/32 | 1/32 |
| GRID redraws with a misplaced range | 0/32 | 5/32 |
| GRID original-time, visible-motion-compatible redraws correctly localized | 0/7 | 0/7 |
| GRID ordinary views flagged | 0/56 | 8/56 |
| H.264 recompressed originals flagged | 0/8 | 2/8 |
| Genuine-picture hard cuts flagged | 0/4 | 3/4 |
| CREMA-D neural edits correctly localized | 1/20 | 2/20 |
| CREMA-D neural edits with a misplaced range | 2/20 | 3/20 |
| CREMA-D ordinary views flagged | 8/40 | 16/40 |

Both arms assessed all 160 views. The one correct GRID range was the same case for both arms, an earlier-driven expression redraw. The extra CREMA-D localization also came from an earlier-conditioned render. The sampled shot guard recognized boundaries in one hard-cut control. The candidate falsely flagged the other three, whose shot boundaries the guard missed. The candidate also flagged ordinary recordings and local resampling, blur and brightness controls. The study required improvements on both banks, including two visibly source-time-compatible GRID cases, with bounded additional ordinary ranges and zero added hard-cut ranges. The candidate missed those requirements.

An independently replayed candidate range was **1.120–1.680 s** on a known **1.080–1.880 s** redraw, with 0.70 IoU. The before, during and after pictures came from **1.040**, **1.360** and **1.680 s** on the exact source clock. The audit verified 156 native frame/crop references and all 160 input file digests. Both arms returned identical observations on the eight matched original/audio-only pairs. The held CPU worker checked 5,498 sampled pictures in 81.5 seconds with 131.4 MiB peak reported RSS. The exact images, source records and model fitting data stay in the private archive.

## Reproduce

Use the retained native source archive and a fresh private output directory. The controls are prepared before held scoring. The independent reviewer recomputes the decisions and checks selected crop pixels and timestamps against the actual source files.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_mouth_jaw_transfer freeze "$run_dir" --archive "$archive"
backend/.venv/bin/python -B -m scripts.video_mouth_jaw_transfer prepare "$run_dir"
backend/.venv/bin/python -B -m scripts.video_mouth_jaw_transfer development "$run_dir"
backend/.venv/bin/python -B -m scripts.video_mouth_jaw_transfer run "$run_dir"
backend/.venv/bin/python -B -m scripts.review_video_mouth_jaw_transfer "$run_dir"
```
