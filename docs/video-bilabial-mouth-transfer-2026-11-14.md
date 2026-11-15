# Locating local lip-sync shifts with visible lip releases

Comparing visible lip openings with speech timings before and after a
suspected mismatch located five of 128 staged shifts. The current
audio–picture timing check located zero on the same clips. The new
check stays in local evaluation. It lacked supporting measurements on
114 staged shifts and issued four flags that failed the position or
direction requirement.

## Method

The two [speech-timestamp cohorts](video-neural-phone-clock-2026-11-14.md)
include 96 short recordings with licensed frontal video, publisher
phone labels, and separate 16 kHz audio. Each of 64 held recordings
from 16 speakers provides two local 200 ms soundtrack shifts in a
700 ms edit window, two constant 200 ms shifts, an unchanged version,
a low-pass version, and a local gain reduction. Each delivery keeps
the exact original pictures. The 448 held deliveries include 192
ordinary inputs, 128 constant delays and 128 local shifts. The cohort
sources were already evaluated for audio timestamp accuracy.
[Corpus and CC BY 4.0 license](https://zenodo.org/records/3736465)

A visual measurement locates a reopening after two low inner-lip
aperture readings measured on consecutive native frames. It uses the
existing pinned face landmark runtime and records the exact source
presentation time, full frame hash, crop box and crop hash. It receives
no soundtrack, phone labels or splice locations. The acoustic stage
reads unprompted /b/ and /p/ events from each waveform. An event needs
a visibly distinct nearest reopening relative to the acoustic clock.
[Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
· [MediaPipe license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)
· [Neural acoustic checkpoint and license](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4)

Four development speakers in the first cohort supply a fit for the
phone-minus-picture time difference using publisher labels. The neural
fit used 11 events from two of those speakers. Four different development
speakers in the second cohort supply separate operating points from
their highest ordinary-input scores. The version requiring context on
both sides uses two aligned events to estimate a local baseline.
The held comparison also calculates the existing timing check with the
same source frames and PCM. A correct localization requires at least
0.40 temporal intersection over union and a signed offset within 80 ms.
These settings measure a timing pattern.

## Held comparison

| Check | Shifts with a usable measurement | Local flags | Correct locations | Ordinary flags | Constant-shift flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current app timing arithmetic | 51/128 | 0 | 0 | 0/192 | 0/128 |
| Earlier speech decoder with context on both sides | 1/128 | 1 | 1 | 0/192 | 0/128 |
| Neural decoder with context on both sides | 14/128 | 9 | 5 | 0/192 | 0/128 |
| Neural decoder with a fixed baseline, tighter point | 122/128 | 0 | 0 | 0/192 | 0/128 |
| Neural decoder with a fixed baseline, broader point | 122/128 | 5 | 2 | 0/192 | 6/128 |

The two tested points gave identical results for both checks requiring
context on both sides. The neural check had sufficient context on
40 of 192 ordinary deliveries and zero of 128 constant delays. On
the nine local flags, two failed the offset direction and two failed
the overlap rule. The five correct locations came from three speakers.
On one shifted clip, the range was **0.733–1.146 s**, the measured
offset was **205 ms earlier**, and the recorded lip opening used
native frames at **1.045 s** and **1.087 s**. The known splice spans
0.695–1.395 s with an offset of 200 ms earlier.

These short scripted clips test a controlled soundtrack edit. Natural
speech, variable rates, multiple speakers, unrelated speakers and
generated mouth movements need their own evaluation. No app report or
AI-confidence calculation uses this cue.

## Replay

The [measurement receipt](evaluation-panels/video-bilabial-mouth-transfer-2026-11-14.json)
stores numeric readings and source commitments. Standard-library Python
recounts decisions and validates the parent receipts. Native replay
regenerates all 672 waveforms, checks 5,847 native presentation times
and verifies the exact pixels for 86 selected frames and mouth crops.
This replay does not rerun recognition. Source recordings and model
weights are available from their publishers.
[Native replay receipt](evaluation-panels/video-bilabial-mouth-transfer-source-replay-2026-11-14.json)

```sh
export PYTHONPATH=backend
first=docs/evaluation-panels/video-neural-phone-clock-2026-11-14.json
second=docs/evaluation-panels/video-neural-phone-context-2026-11-14.json
receipt=docs/evaluation-panels/video-bilabial-mouth-transfer-2026-11-14.json
python3 -B -S -m scripts.video_bilabial_mouth_transfer review "$receipt" --parents "$first" "$second"
backend/.venv/bin/python -B -m scripts.video_bilabial_mouth_transfer replay "$receipt" \
  --parents "$first" "$second" --sources "$sources" --output "$new_replay"
```

For a full rerun, prepare the originals, acoustic weights and decoder
wheel as described in the [parent test](video-neural-phone-clock-2026-11-14.md#independent-replay).
Use MediaPipe 0.10.32 and the face task pinned in the evaluation script.
Run `observe "$new_folder" --parents "$first" "$second" --sources "$sources"
--runtime "$mediapipe_runtime" --geometry "$face_task" --checkpoint "$checkpoint"
--wheel "$decoder_wheel"` followed by `analyze "$new_folder/observed.json"
--parents "$first" "$second" --output "$new_receipt"` in the project
Python environment. The reported observation used Apple Metal.
