# Short gaps in speech activity under room sound

This comparison tests whether a drop in measured speech activity can locate
missing speech while the room sound and a visible person's mouth movement
continue. The finding concerns the timing of activity. Speaker identity,
voice generation and the cause of the decline are outside its scope.

## Frozen comparison

The method reads speech activity in native 512-sample blocks on the exact
16 kHz soundtrack clock. It compares the center of each candidate interval
with two 160 ms flanks whose mean activity is at least 0.35. Each candidate
also needs mouth motion and source frame anchors before, during and after
the interval. The three readings use the original stereo
mid-channel, the same audio after noise reduction, or the smaller activity
drop when both views qualify for the same interval. Noise reduction preserves
sample count. Any report would use original audio and pictures.

The published speech-activity weights are MIT licensed. The noise reducer
uses the separately pinned official MIT release and DNS3 checkpoint from the
repository's prior license review. The study uses no added weights.
[Silero release and license](https://github.com/snakers4/silero-vad/tree/066409f3d9db45a30d21dfcbf9147826ffdd7052) ·
[GTCRN release](https://github.com/Xiaobin-Rong/gtcrn/tree/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73) ·
[Project license review](audio-noise-view-v4-development.md#question-and-source-permission)

Recent speech-activity research compares model behavior at its trained block
size and measures speech transition timing separately from ranking. Here each
view is judged directly against the known edit range. [Window Size Matters](https://arxiv.org/html/2610.14607v2) ·
[S4VAD](https://arxiv.org/abs/2609.11110)

The 280 original development deliveries set separate operating points for
PCM and AAC using the same speaker-group budgets that preceded the phase
comparison. The fresh held panel reserves different speakers, source files and
room takes from both earlier stereo panels. Each uses two staged removals and
five ordinary stereo controls at each foreground level and encoding. Each
timed observation is scored at a sample-range intersection over union of 0.4
or more. The two encodings are paired and share edit truth.

A candidate qualifies for a separate app replay if it gains at least three
correct loud-background locations across at least two speakers per encoding
over the stronger prior reading, while adding at most three ordinary alerts
per encoding and at most two ordinary speaker groups. Review both background
levels for regression. A qualified result can support a scoped inspection
cue with zero AI-confidence weight. Its measured error rates apply to these
staged edits. The next check would verify the displayed sample and picture
positions and any expected distinction from ordinary pauses in speech.

## Held results

The frozen joint reading found four additional loud-background locations
from two speakers for each encoding. It kept both previous correct locations
in its new total. The previous stereo comparison flagged nine of 60 ordinary
loud-background clips. The new reading flagged one. The selection also
improved localization where the voice was louder than the room. The single
loud-background alert came from a different speaker group. The new reading
missed one quieter-room edit previously located in each encoding.

| Voice relative to room | Encoding | Correct, previous stereo → joint | Wrong range on edited clip, previous → joint | Ordinary clips flagged, previous → joint |
| --- | --- | ---: | ---: | ---: |
| 5 dB below | PCM | 2 → 6 of 24 | 2 → 2 of 24 | 9 → 1 of 60 |
| 5 dB below | AAC | 2 → 6 of 24 | 2 → 1 of 24 | 9 → 1 of 60 |
| 8 dB above | PCM | 6 → 15 of 24 | 0 → 2 of 24 | 0 → 2 of 60 |
| 8 dB above | AAC | 5 → 15 of 24 | 0 → 2 of 24 | 0 → 2 of 60 |

The near-silence cue returned zero ranges. The new reading could compare 18
of 24 edited clips under loud room sound, and 22 of 24 under quieter room
sound. It missed 12 of the 18 assessed loud edits. The 12 held actors had
appeared in earlier timing work. Those exact clips and the speakers and
room recordings used in both earlier stereo studies were excluded.
[CREMA-D source and license](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) ·
[DCASE scene recordings](https://archive.org/details/dcase2013_scene_classification)

## Local analysis

Set `VERITAS_VIDEO_SPEECH_ACTIVITY_GAP_ENHANCER` to the pinned noise-reducer
folder. The check uses the local speech-activity checkpoint and the existing
`VERITAS_VIDEO_LIP_TIMING_FACE_ASSET` and, when needed,
`VERITAS_VIDEO_LIP_TIMING_RUNTIME` paths. It runs with Full analysis in a local
development environment on continuous 30 fps single-shot videos with one
stereo soundtrack and a verified zero-origin clock. The tested carriers are
16 kHz PCM and 48 kHz AAC-LC. Clearing the enhancer setting rolls back the
option. A trusted AI provenance record skips the statistical review.

The web view links the interval to the same waveform, playable excerpt,
original before/during/after frames and mouth crops used in the report.
JSON records the exact 16 kHz sample interval and original picture timestamps.
The waveform and excerpt come from the submitted channels. Original-file
replay re-decodes both channels, checks the stereo digest, checks the shot
and native mouth crops, and recreates the excerpt and waveform. The retained
activity measurements reproduce the selected range during replay.

The local AAC and PCM runs both reported **0.832–1.344 s**, with original
source frames **24, 33 and 41**. Eight assets per result matched their
published digests, and the PDF retained the original pictures, mouth crops
and waveform. The JSON export matched the signed package's evidence.
The browser decoded all seven images and loaded the original-channel mix.
The unchanged video yielded no activity range, and switching the option off
preserved the automated assessment. Each of the four simulated USD 9.99
payments queued one analysis after payment and zero before.
[Local checkout and browser receipt](evaluation-panels/video-speech-activity-gap-app-2026-11-17.json)

## Replay

The [development](evaluation-panels/video-speech-activity-gap-development-2026-11-17.json.gz)
and [held](evaluation-panels/video-speech-activity-gap-held-2026-11-17.json.gz)
receipts retain the operating points, coverage, edit intervals and actual
decisions. The [source reservation](evaluation-panels/video-speech-activity-gap-plan-2026-11-17.json.gz),
[publisher inventory](evaluation-panels/video-speech-activity-gap-sources-2026-11-17.json.gz)
and [held inputs](evaluation-panels/video-speech-activity-gap-inputs-2026-11-17.json.gz)
bind all 336 deliveries. Media and the complete coordinate traces stay local.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
python3 -B -S -m scripts.video_speech_activity_gap_panel verify \
  "$panels/video-speech-activity-gap-held-2026-11-17.json.gz" \
  --freeze "$panels/video-speech-activity-gap-development-2026-11-17.json.gz"
```

Rebuild the reservation with `scripts.video_speech_dropout_sources` and use
the `download` and `prepare` commands in `scripts.video_covariance_sources`
on that reservation. These use the pinned CREMA-D metadata and DCASE stereo
archive documented for the [previous comparison](video-stereo-covariance-2026-11-17.md#reproduce).
Use `scripts.video_speech_activity_gap_panel measure --help` for an exact-file
measurement using the prescribed activity and noise-reduction assets.
Use `scripts.video_speech_activity_gap_app --help` to replay the four local
checkouts, PDF and JSON exports, and browser verification.

A separate standard-library recount verified 616 delivery hashes, the source
reservation and freeze, both carrier comparisons, and the mouth frame and
sample brackets. It reproduced gains and losses by speaker.
