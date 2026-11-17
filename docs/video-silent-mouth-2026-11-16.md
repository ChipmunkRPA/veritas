# Mouth movement during a near-silent video soundtrack

Chipmunk Verify can now highlight a brief drop to near-silence while the
visible mouth keeps moving. The finding shows the source time, the selected
waveform, the original excerpt, and native frames and mouth crops from
before, during and after the interval. The finding has zero weight in the AI
assessment. Muting, a recording dropout or quiet movement can produce it.

## What the application checks

Set `VERITAS_VIDEO_LIP_SILENT_MOUTH_REVIEW=true` on a development backend
with the local speech and mouth review enabled and its existing assets
available. The check runs on the Full profile for eligible continuous
shots. It accepts mono 16 kHz PCM and 48 kHz AAC with a verified presentation
clock. The existing speech and mouth checks take priority if they have
already found a range. The fallback compares 5 ms blocks and requires an
interior interval of 150–800 ms with a peak of at most four units of
16-bit amplitude, audible sound on both sides, and changing mouth opening
across at least four native frames. AAC padding stays outside the measured
range. The frozen settings are reproducible measurement conditions.
[Pinned local assets](video-bilabial-mouth-transfer-2026-11-14.md)

This check measures near-silence even if it contains a small residual signal.
Speech replaced by other speech, a mute with substantial background sound,
and generated or edited mouth motion require separate comparisons.

## Comparison on fresh staged edits

The panel uses 36 previously unused Lombard Grid recordings from 18 speakers.
Each recording has a hard mute, a mute with 7.5 ms fades, and seven control
deliveries. The mute intervals are 240, 360 or 520 ms long. The test preserves
source pictures and uses the same edited sound in PCM and 48 kHz AAC.
The controls include unchanged audio, ordinary gain, 90% local level
reduction, same- and different-word replacement, and whole-track shift in
either direction. A correct interval overlaps the known range with temporal
intersection over union of at least 0.40. The deliveries reuse recordings.
[Lombard Grid recordings and CC BY 4.0 terms](https://zenodo.org/records/3736465)

| Outcome | PCM, existing → with new check | 48 kHz AAC, existing → with new check |
| --- | ---: | ---: |
| Hard mute correctly located | 3 → 30 of 36 | 3 → 30 of 36 |
| Faded mute correctly located | 2 → 27 of 36 | 2 → 27 of 36 |
| Any range on an out-of-scope control | 20 → 20 of 252 | 20 → 20 of 252 |

The new finding added 104 correct locations and no flags on 504 out-of-scope
controls. The earlier speech and mouth review still accounts for the 40
existing ranges on those controls. All ranges and individual dispositions are
included in the measured receipt. The results describe the panel's short,
single-speaker clips with visible mouths.
[Measured sample ranges](evaluation-panels/video-silent-mouth-2026-11-16.json.gz)
· [Confirmation source plan](evaluation-panels/video-lip-donor-confirmation-sources-2026-11-16.json.gz)

In the prepaid local app replay, the edited AAC clip showed the interval
**1.080–1.290 seconds**. Chromium loaded the waveform, excerpt and six
pictures. The PDF, JSON and signed package carried the same range, frame
anchors and exhibit contents. Replaying with the source video reproduced
the audio and frame assets. The unmodified clip and the edited clip with the
option disabled showed no near-silent finding. Turning the check on left the
AI assessment for the edited file unchanged. All three simulations queued
zero jobs before a paid event for USD 9.99.
[Application receipt](evaluation-panels/video-silent-mouth-app-2026-11-16.json)

## Comparing a separate word-replacement candidate

A second experiment trained a small comparison on mismatched words from
the same speaker and recording condition. Two arms used mouth geometry or
geometry with appearance measured from the original mouth crops. The fit
used 168 utterances from 14 speakers. A separate seven-speaker set picked
the geometry arm. The 24 earlier held recordings added one correct
location for each carrier. On the 36 fresh recordings above, the frozen
arm gave these results for distinct-word localization.
[Frame-level speech and mouth correspondence research](https://arxiv.org/html/2610.30872v1)
· [Local articulation guidance](https://arxiv.org/html/2610.09223v1)

| Outcome | PCM, existing → with candidate | 48 kHz AAC, existing → with candidate |
| --- | ---: | ---: |
| Correct distinct-word location | 5 → 7 of 36 | 6 → 8 of 36 |
| Untouched or gain-only flags | 3 → 7 of 72 | 3 → 7 of 72 |
| Same-word flags | 3 → 5 of 36 | 3 → 5 of 36 |
| Whole-track shifts with a local flag | 4 → 11 of 72 | 3 → 9 of 72 |

The candidate remains in evaluation because the measured gains came with
more flags on ordinary audio and whole-track shifts. The application uses
the existing word and mouth comparison.
[Pinned fit](evaluation-panels/video-lip-donor-fit-2026-11-16.json.gz)
· [Earlier held results](evaluation-panels/video-lip-donor-contrast-2026-11-16.json.gz)
· [Fresh confirmation](evaluation-panels/video-lip-donor-confirmation-2026-11-16.json.gz)

## Reproduce the measurements

The source plan pins the archive recordings used to author each variant.
The playback manifest verifies 648 cases and 1,296 source and delivery file
bindings. Counting on the native picture grid reproduces the sample-based
location totals. The repository includes compact measurements and source
pointers. Rebuild
the videos from the publisher's audio and frontal video archives into a
local review directory. Native capture additionally needs the pinned face
and speech assets. [Source and asset preparation](video-neural-phone-clock-2026-11-14.md#independent-replay)
· [Portable inputs, truth and event counts](evaluation-panels/video-silent-mouth-playable-2026-11-16.json.gz)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
panels=docs/evaluation-panels
plan="$panels/video-lip-donor-confirmation-sources-2026-11-16.json.gz"
mute="$panels/video-silent-mouth-2026-11-16.json.gz"
fit="$panels/video-lip-donor-fit-2026-11-16.json.gz"
donor="$panels/video-lip-donor-confirmation-2026-11-16.json.gz"
python3 -B -S -m scripts.review_video_speech_mismatch silence "$mute"
python3 -B -S -m scripts.review_video_speech_mismatch donor "$donor"
"$runner" -B -m scripts.export_video_silent_mouth "$plan" "$mute" "$sources" "$run/playable"
python3 -B -S -m scripts.independent_review verify-inputs \
  "$run/playable/manifest.json" "$run/playable/file-index.json" "$run/readback.json" --root "$run/playable"
python3 -B -S -m scripts.independent_review count-events \
  "$run/playable/manifest.json" "$run/playable/truth.json" "$run/playable/combined.json" "$run/events.json" --match .40 .40
"$runner" -B -m scripts.video_silent_mouth_panel "$plan" "$sources" "$assets" "$run/silence"
"$runner" -B -m scripts.video_lip_sequence_panel capture "$plan" "$sources" "$assets" held "$run/donor-input.json.gz"
"$runner" -B -m scripts.video_lip_donor_contrast assess "$fit" "$plan" "$run/donor-input.json.gz" "$run/donor-result.json.gz"
```

For the application replay, build the frontend and run
`scripts.video_silent_mouth_app` with `$run/silence`, the pinned asset
directory, and a new output directory. It uses a separate data directory,
a simulated payment, and Chromium. To refit the experimental geometry and
appearance arms, capture `fit`, `calibration` and `held` with the
[parent source plan](evaluation-panels/video-lip-sequence-sources-2026-11-16.json.gz)
into `fit.json.gz`, `calibration.json.gz` and `held.json.gz` alongside
`plan.json.gz`. Pass that directory and a new output directory to
`scripts.video_lip_donor_contrast fit`.
