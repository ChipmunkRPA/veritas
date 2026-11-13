# Single-upload mouth color flicker and agreement

A November 2026 preprint found that frame-to-frame color in several deepfake
benchmarks can encode video source and generator processing. Its simple
chrominance classifier trained on one corpus marked more than 99% of real
videos from another corpus as fake. We tested a narrower claim. A useful cue
would need to localize a short redraw from the change in nearby regions of
the same recording. [Harraq and Aldwairi, *Pulse or Provenance?*](https://arxiv.org/abs/2611.13992)

## Frozen design

The check tracks five small skin areas in an eye-aligned plane. It uses two
lower-face patches on either side of the lips, two cheek controls and a common
upper-face patch. Strong gradients and clipped values are excluded. Color
changes are measured by differencing each patch against the upper face, then
taking centered temporal second differences over one and two native frames.
The first arm asks whether lower-face chroma flickers more than cheek chroma
relative to local luminance. The second asks whether left and right lower-face
colors disagree more than the cheeks. Neither calculation estimates a pulse.
The only model is the project's pinned MIT-licensed face geometry. Earlier
copies and audio do not enter either arm. [Face geometry license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

The 8-, 12-, 16- and 20-picture candidate ranges require six-picture flanks,
one observed shot and measured support at both temporal distances. The third
largest score on 56 GRID ordinary controls fixes the operating point for each
arm, subject to the predeclared floor. The held comparison reuses 92 H.264
deliveries. They include 17 authored redraws across two workflows and 75
ordinary controls. Eleven redraws retained an independently measured local
speech timing event. The source material had prior project exposure.
[Panel construction and terms](video-mouth-blind-prediction-2026-11-13.md)

## Held results

| Outcome | Relative color flicker | Left-right disagreement |
| --- | ---: | ---: |
| Correct redraw ranges at the frozen point | 0/17 | 0/17 |
| Correct redraw ranges before the operating point | 4/17 | 3/17 |
| Correct ranges in the timing-qualified subset | 0/11 | 0/11 |
| Redraws flagged at an incorrect time | 1 | 0 |
| Ordinary clips flagged | 8/75 | 0/75 |
| Clips measured | 92/92 | 92/92 |

Four untouched clips and one ordinary hard cut account for five of the eight
flicker flags. A delayed-pixel delivery, a same-time resample and a temporal
blend account for the other three. Neither arm enters the app or the AI
confidence calculation. These results apply to the prepared short H.264
panel and leave generalization to longer video and other generators open.
The native reader independently reopened all 27 RGB frames and crops across
the nine flags. [Decision ledger](evaluation-panels/video-mouth-temporal-chroma-2026-11-13.json)

## Replay

```sh
python3 -B -IS backend/scripts/review_video_mouth_temporal_chroma.py \
  docs/evaluation-panels/video-mouth-temporal-chroma-2026-11-13.json
```

Use the existing native H.264 study directory as `parent` for a fresh scoring
run. The scorer requires the existing pinned geometry weights and FFmpeg.
Pass `--carriers "$parent/carriers.json"` to the reader without `-I` to replay
flagged native anchors and source crops.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.video_mouth_temporal_chroma "$phase" "$parent" "$run" || exit 1
done
```
