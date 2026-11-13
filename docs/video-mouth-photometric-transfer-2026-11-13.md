# Single-upload mouth-region color and texture test

The test measures changes in skin on both sides of the mouth relative
to nearby cheek and upper-face patches. It excludes the central lips,
strong gradients and clipped pixels. The second arm requires a loss of
local fine texture at the same time. A single upload supplies the pixels,
face landmarks and picture clock. The method uses neither the earlier
copy nor speech. This probe follows research that treats blending and
illumination as separate hypotheses, and tests whether a lighting cue adds
value under ordinary changes. [Face-swap illumination study](https://arxiv.org/abs/2610.11706)
· [Localized facial motion study](https://openaccess.thecvf.com/content/CVPR2025W/SAIAD/papers/Anand_Detecting_Localized_Deepfake_Manipulations_Using_Action_Unit-Guided_Video_Representations_CVPRW_2025_paper.pdf)

## Frozen comparison

The check searches 8-, 12-, 16- and 20-picture intervals with five
pictures on either side in the same shot. The third-highest response in
56 ordinary GRID deliveries set a separate internal point for each arm,
subject to a fixed floor. The held set reused 92 exact H.264 deliveries,
including 17 authored redraws, 75 ordinary controls and two synthesis
families. Eleven of the redraws had an independently observed timing
event. The ordinary set includes untouched clips, conventional local
resampling, retimes, recompression and genuine cuts. The clips had
previous project exposure. No new detector weights were loaded beyond
the reviewed face geometry checkpoint. [Panel and source terms](video-mouth-blind-prediction-2026-11-13.md)
· [Face geometry license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

| Held outcome | Color contrast | Color and texture |
| --- | ---: | ---: |
| Correct redraw intervals before applying point | 2/17 | 2/17 |
| Correct redraw intervals at frozen point | 2/17 | 1/17 |
| Timing-qualified redraws located | 2/11 | 1/11 |
| Ordinary alerts | 29/75 | 0/75 |
| Source deliveries measured | 92/92 | 92/92 |

Neither arm located any of the six held redraws from the second
synthesis family. Seven of the color-only ordinary alerts came from
untouched originals or genuine hard cuts. The corroborated arm's
sole hit covered 0.751–1.084 seconds against an authored range of
0.667–1.335 seconds, exactly half the authored range. The texture
requirement suppressed ordinary flags but did little to recover
the timing gap. Neither arm passed the frozen rule or entered serving.
[Decision ledger](evaluation-panels/video-mouth-photometric-transfer-2026-11-13.json)

## Replay

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.video_mouth_photometric_transfer "$phase" "$parent" "$run" || exit 1
done
python3 -B -IS backend/scripts/review_video_mouth_photometric_transfer.py \
  docs/evaluation-panels/video-mouth-photometric-transfer-2026-11-13.json
```

The decision reader requires no media. If the retained H.264 files are
available, pass `--carriers "$parent/carriers.json"` to the reader without
`-I` to independently verify each flagged frame and its exact source crop
with FFmpeg. The native replay verified 111 crops from 36 flagged files.
The crop identifies the inspected region. It does not
represent an authored edit mask.
