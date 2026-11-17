# Short differences between visible mouth movement and speech

Chipmunk Verify can flag a short region where the visible mouth and the sound
agree less than in the surrounding speech. The comparison uses a close view
of the mouth and the delivered audio. It scans the original picture times
and reports the audio range, three original frames and mouth crops, and a
waveform and playable excerpt. It carries zero weight in the AI assessment.
Dubbing, offscreen speech and ordinary differences in pronunciation or timing
can affect the comparison.

## Method and calibration

Frame-level lip and audio research motivates comparing short local sequences
whose overall pacing may agree. This implementation uses a shared
representation learned from genuine synchronized recordings. Each picture
reading combines the same face-aligned mouth features from five consecutive
frames. The audio path uses a short log-frequency spectrum around each time.
Fitting compares genuinely aligned observations with shifted positions in
the same clip and another recording by the same speaker. A separate arm
adds broad speech-sound labels during fitting. Neither arm needs a
transcript or a known word time during an analysis.
[Frame-level speech and mouth research](https://arxiv.org/html/2610.30872v1)
· [Timing and articulation research](https://arxiv.org/html/2610.09223)

The fit used 16,145 paired time positions from 336 genuine Lombard Grid
recordings of 14 speakers. The face-landmark runtime is the previously
reviewed local asset. The learned weights use these licensed source
recordings and project code. No paper checkpoint is used. 41 recordings
from seven separate speakers set the decision points by source-group maxima.
The codec-specific stricter points come from the largest new control score.
[Lombard Grid licence and source](https://zenodo.org/records/3736465)
· [Landmark and baseline asset review](video-lip-articulation-2026-11-15.md)
· [Training receipt](evaluation-panels/video-mouth-sound-fit-2026-11-17.json.gz)
· [Frozen decision points](evaluation-panels/video-mouth-sound-selection-2026-11-17.json.gz)

One common alignment must retain enough agreement on both sides of a
candidate. The comparison allows nearby timing inside the range and
suppresses a local proposal where a consistent delay better explains both
halves of the clip. The current event-based review has priority when it
already has a source-bound finding.

## Matched video trials

The first held panel uses 24 unused target and donor sets from 24 speakers.
The GRID panel uses 16 original recordings from a separate capture setup.
The final confirmation uses 18 unused sets from nine Lombard speakers
outside this comparison's fitting, selection and first held panel. Neither
their target nor donor recordings repeat earlier word-comparison inputs.
All test cases replace a digit with a duration-matched same-speaker recording.
Controls retain the original audio, change gain, use a different rendition of
the same digit, or shift the whole track by ±200 ms. A useful highlight needs
at least 0.40 temporal intersection over union with the known changed
sample range. PCM and AAC deliveries share the same source groups. All
696 deliveries reached the comparison.
[GRID source and licence](https://zenodo.org/records/3625687)
· [Initial held source plan](evaluation-panels/video-mouth-sound-held-sources-2026-11-17.json.gz)
· [Final confirmation source plan](evaluation-panels/video-mouth-sound-confirmation-sources-2026-11-17.json.gz)

| Panel and encoding | Correct edit locations, current → combined | New flags on untouched or gain-only uploads | New flags on same-word replacements | New flags on whole-track delays | Misplaced new edit locations |
| --- | ---: | ---: | ---: | ---: | ---: |
| First Lombard · PCM | 2 → 6 of 24 | 2/48 | 2/24 | 3/48 | 1 |
| First Lombard · AAC | 2 → 6 of 24 | 2/48 | 2/24 | 3/48 | 1 |
| Separate GRID · PCM | 0 → 1 of 16 | 0/32 | 1/16 | 1/32 | 1 |
| Separate GRID · AAC | 0 → 2 of 16 | 1/32 | 1/16 | 1/32 | 1 |
| Final confirmation · PCM | 0 → 2 of 18 | 1/36 | 2/18 | 0/36 | 1 |
| Final confirmation · AAC | 0 → 2 of 18 | 2/36 | 2/18 | 0/36 | 1 |

The stricter shared representation locates seven additional edits in
PCM and eight in AAC across the three panels. Ordinary files with a
highlight increase from 4 to 7 of 116 for PCM and from 4 to 9 of 116
for AAC. The additional ordinary highlights come from two PCM and
three AAC speaker groups. The wider points add up to 14 ordinary
highlights on the first panel alone. An alternative fixed comparison
requires agreement between both learned readings. It adds five
locations per encoding across the first two panels and zero on
confirmation. The local option uses the stricter single reading.
[Held measurements](evaluation-panels/video-mouth-sound-held-2026-11-17.json.gz)
· [GRID measurements](evaluation-panels/video-mouth-sound-grid-2026-11-17.json.gz)
· [Final measurements](evaluation-panels/video-mouth-sound-confirmation-2026-11-17.json.gz)
· [Two-reading comparison](evaluation-panels/video-mouth-sound-two-readings-2026-11-17.json.gz)

The tests stage edits using genuine speech in controlled recordings.
They measure disagreement between sound and mouth movement. They leave
ordinary unscripted speech, loud competing speech, videos with
regenerated mouth movement, and videos with multiple speakers
unqualified for this comparison's measured error rates.

## Local analysis and independent replay

The optional check uses a Full analysis on a short, continuous upright
shot near 25 fps. It requires one mono soundtrack, either 16 kHz PCM or
48 kHz AAC, with verified continuous timing. The picture and audio
presentations begin at zero. Enable the existing lip-timing setting and
set `VERITAS_VIDEO_LIP_MOTION_SOUND_MODELS` to the folder containing
`vision.safetensors` and `shared.safetensors`. The worker verifies
both model files before loading them. Returning to the existing timing
behavior only requires clearing the path.

The local prepayment test uses simulated USD 9.99 payments. The ordinary
view links to the supporting frames, mouth crops, waveform and audio.
The PDF and JSON use the same positions and measurements. The signed
replay verifies the source timestamps, picture crops and exact soundtrack
excerpt. Both encodings highlighted audio from 1.149 to 1.358 s and
picture from 1.129 to 1.338 s. Source frames 26, 28 and 32 show the
before, during and after views. The automated assessment stays the same
with the option off.
[Checkout, browser and source replay](evaluation-panels/video-mouth-sound-app-2026-11-17.json)

The numeric reader uses standard-library Python. Native readback rebuilds
504 videos across the two Lombard holdouts. The original frames and
presented audio are independently bound to each generated input.
[First native readback](evaluation-panels/video-mouth-sound-held-native-2026-11-17.json.gz)
· [Final native readback](evaluation-panels/video-mouth-sound-confirmation-native-2026-11-17.json.gz)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
panels=docs/evaluation-panels
calibration="$panels/video-mouth-sound-selection-2026-11-17.json.gz"
for panel in held grid confirmation; do
  python3 -B -S -m app.bench.video_av_local_agreement \
    "$panels/video-mouth-sound-$panel-2026-11-17.json.gz" --freeze "$calibration"
done
```

Rebuild the published visual fit and native crops from the [mouth capture
recipe](video-mouth-patch-speech-2026-11-17.md#reproduce). The fitting
command uses the same verified Lombard publisher ZIPs.

```sh
runner=backend/.venv/bin/python
"$runner" -B -m scripts.video_av_local_agreement fit "$sources" \
  "$visual_fit/visual-speech.safetensors" "$run/fit" \
  "$panels/video-mouth-patch-sources-2026-11-17.json.gz" "$fit_crop"
mkdir -p "$run/local-models"
cp "$visual_fit/visual-speech.safetensors" "$run/local-models/vision.safetensors"
cp "$run/fit/shared.safetensors" "$run/local-models/shared.safetensors"
```

For a held source plan, rebuild the mouth crops and the existing timing
baseline using the [word comparison native capture commands](video-word-content-2026-11-17.md#reproduce).
The `measure` command below accepts the held source plan, its corresponding
mouth crop folder and the measured timing baseline. It re-encodes both
audio carriers and checks the original samples and frame timestamps.

```sh
"$runner" -B -m scripts.video_av_local_agreement measure "$sources" \
  "$run/local-models/vision.safetensors" "$run/held.json.gz" "$run/fit" \
  --freeze "$calibration" --input "$held_plan" "$held_crops" "$held_timing"
"$runner" -B -m scripts.export_video_av_local_agreement \
  "$held_plan" "$calibration" "$run/held.json.gz" "$sources" "$run/native"
npm --prefix frontend run build
"$runner" -B -m scripts.video_lip_motion_app \
  "$run/held.json.gz" "$run/native" "$assets" "$run/local-models" "$run/app" --browser
```

Use the final confirmation plan for the prepaid example. The public receipts
contain source commitments and numerical measurements. Original recordings,
mouth crops, trained weights and customer uploads stay on the machine
running the replay.
