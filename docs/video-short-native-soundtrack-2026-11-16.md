# Short speech replacements on the original video clock

The earlier-video comparison now supports short 720 × 480 H.264 pairs
with identical decoded pictures and equal native picture clocks near
24 fps. It compares the first mono soundtrack from each file, including
pairs delivered as 16 kHz PCM and 48 kHz AAC-LC. It uses the AAC container's
presentation duration to exclude trailing decoder samples from the
comparison. A supported result shows the two original audio excerpts,
their waveforms, and six original pictures around the same time range.
The result describes a soundtrack difference and carries zero AI-score weight.

## Earlier-copy test

The check keeps the existing voice-band measurements and requires matching
sound on either side of the local difference. The app first checks for
ordinary local timing moves. Each supported short video has a constant
native picture step from 23.8 to 24.2 fps and identical timed picture
content in both copies. Audio presentations must start with the picture,
have equal sample counts and span about two to four seconds. The first
trial limited intake to 23.976 fps and excluded 18 of the 24 recordings.
Reviewing source timestamps identified three other exact cadences near
24 fps. Expanding clock admission preserved the original signal settings.
[Existing sound-band method](video-reference-sound-difference-2026-11-14.md)

The new panel has 24 previously unused recordings from 18 Lombard Grid
speakers. A different-word case replaces one letter and one digit with a
duration-matched recording of the same speaker. A same-word case replaces
the corresponding passage with a separate rendition of the same words.
Both are intentional changes to the soundtrack. The controls include an
untouched track, a gain adjustment, and both directions of whole-track
delay. The edited span and unchanged picture bitstream are recorded before
delivery. [Lombard Grid data and CC BY 4.0 terms](https://zenodo.org/records/3736465)

The app comparison ran on all 432 pairs. Each delivery uses the same source
recordings and two forms of the intended edits. A correct location needs
at least 0.40 temporal intersection over union with the known range.

| Outcome per delivery | PCM to PCM | AAC to AAC | PCM to AAC |
| --- | ---: | ---: | ---: |
| Correct different-word replacement locations | 15 of 24 | 15 of 24 | 15 of 24 |
| Correct same-word replacement locations | 15 of 24 | 15 of 24 | 15 of 24 |
| Untouched or gain-only controls with a flagged range | 0 of 48 | 0 of 48 | 0 of 48 |
| Untouched or gain-only controls assessed | 46 of 48 | 46 of 48 | 46 of 48 |
| Different-word replacements unassessed | 8 of 24 | 8 of 24 | 8 of 24 |

One source remained outside the picture profile. Seven edited clips lacked
sufficient matching sound in the flanks, and one assessed edit produced no
qualifying range in each delivery. All 48 whole-track delays per delivery
stayed unassessed by the local difference check. A missing range leaves the
comparison unresolved. Background noise and broader camera formats need
separate measurements. [Paired outcomes and source clocks](evaluation-panels/video-short-native-soundtrack-2026-11-16.json.gz)

In development, set `VERITAS_VIDEO_REFERENCE_SHORT_NATIVE_SOUND_DIFFERENCE=true`
with the earlier-video and soundtrack comparison options. In a completed
analysis, choose **Compare with an earlier video** and select **Include
experimental soundtrack comparison**. The tested AAC replacement shows the
difference from **1.400 to 1.720 s**, two 0.320-second excerpts, and six native
pictures. The app replay uses two simulated USD 9.99 checkouts. Zero jobs
queued before payment or during the later comparison. The standalone PDF
contains the range and shared presentation, and JSON replay re-decodes both
original files and verifies every exhibit byte. [Browser and PDF replay](evaluation-panels/video-short-native-soundtrack-app-2026-11-16.json)

## Single-upload comparison

A separate experiment compared short ordered mouth measurements with short
ordered acoustic sequences. A small model learns to distinguish real
synchronized pairs from shifted pairs taken from the same 168 recorded
utterances and 14 speakers. One arm adds a brightness and color grid from
the native mouth crops. The comparison uses one common time offset on
each side of the candidate range. Seven separate calibration speakers across
13 recordings selected the geometry comparison at its stricter setting.
The project fits its own small model and uses the existing rights-reviewed
assets for mouth observations and the incumbent comparison. [Frame-level audio and lip interaction research](https://arxiv.org/html/2610.30872v1)
· [Pinned speech checkpoint terms](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4)
· [Face-landmark asset and runtime](video-bilabial-mouth-transfer-2026-11-14.md)

| Held results with the candidate added after the current check | PCM | 48 kHz AAC |
| --- | ---: | ---: |
| Useful different-word locations | 3 → 4 of 24 | 3 → 5 of 24 |
| Untouched or gain-only flags | 0 → 0 of 48 | 0 → 0 of 48 |
| Same-word flags | 2 → 5 of 24 | 2 → 6 of 24 |
| Whole-track delay flags | 1 → 3 of 48 | 0 → 2 of 48 |

The gains are smaller than the additional same-word or delay flags. The
single-upload candidate remains experimental. The held recordings all use
short scripted speech and a visible frontal mouth. Transfer to free speech,
generated voices and regenerated mouth motion remains open. [Single-upload results](evaluation-panels/video-lip-sequence-2026-11-16.json.gz)

## Independent replay

The source manifest and event counts bind 288 playable videos and 576 source
and delivery files. The media reconstruction reproduces every held picture
bitstream and timestamp and each presented audio sample. Recounting over the
original picture grid agrees with the single-upload location counts above.
The repository holds the source pointers and measured ranges. Reconstructed
media and dense recordings of features stay in the reviewer's local workspace.
[Source plan](evaluation-panels/video-lip-sequence-sources-2026-11-16.json.gz)
· [Portable manifest and measurements](evaluation-panels/video-lip-sequence-playable-2026-11-16.json.gz)

Place the publisher's `lombardgrid_audio.zip` and `lombardgrid_front.zip`
in the local source directory. Use the published feature and checkpoint
instructions for the fit and holdout capture. The playable export uses the
published result directly and can run without inference or landmark weights.
The local media readback used FFmpeg 7.0.2. [Pinned source and asset preparation](video-neural-phone-clock-2026-11-14.md#independent-replay)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
panels=docs/evaluation-panels
plan="$panels/video-lip-sequence-sources-2026-11-16.json.gz"
result="$panels/video-lip-sequence-2026-11-16.json.gz"
python3 -B -S -m scripts.video_lip_sequence_panel verify "$result"
python3 -B -S -m scripts.video_short_native_soundtrack_panel verify "$panels/video-short-native-soundtrack-2026-11-16.json.gz"
"$runner" -B -m scripts.export_video_lip_sequence "$plan" "$result" "$sources" "$run/playable"
python3 -B -S -m scripts.independent_review verify-inputs \
  "$run/playable/manifest.json" "$run/playable/file-index.json" "$run/readback.json" --root "$run/playable"
python3 -B -S -m scripts.independent_review count-events \
  "$run/playable/manifest.json" "$run/playable/truth.json" "$run/playable/combined.json" "$run/events.json" --match .40 .40
"$runner" -B -m scripts.video_short_native_soundtrack_panel run "$run/playable" "$result" "$run/paired.json.gz"
```

For a full training replay, pass the published plan and the pinned asset
directory to `scripts.video_lip_sequence_panel capture`. Capture `fit` and
`calibration` before `calibrate`, then capture `held` and run `assess` with
the frozen fit. Run `scripts.video_short_native_soundtrack_app` against the
earlier 19-clip benchmark's `clips` directory to repeat checkout, PDF and
JSON replay. Build the frontend and add `--browser` to verify images and
playback in Chromium. [Earlier media reconstruction](video-lip-correspondence-2026-11-16.md#rebuild-the-playable-benchmark)
