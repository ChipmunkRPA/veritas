# Native source inserts with known time boundaries

This data-only construction extends the [completed source qualification](video-native-source-v92-result.md).
It prepares controlled visual-localization inputs, not detector results or an
already admitted model. The 336 original native assets remain unchanged.

## Source eligibility and fixed construction

Keep all 96 planned face-swap source groups recorded. Ninety have equal native
picture times and dimensions across the original and both swaps, with an exact
constant observed clock. Retain the other six as incompatible; do not match
their equal-numbered frames by assumption or replace them after a response.
The known ancestry/component roles remain unchanged. All these sources are
silent; refuse an unexpected soundtrack rather than silently removing it.

For each compatible group, retain its native RGB dimensions and exact rational
frame step. Use the source pictures whose observed presentation times, relative
to the first picture, are under six seconds. This caps picture-center support;
the final encoded display duration need not be exactly six seconds. It is a
controlled excerpt, not a substitute for later full-native-clip comparison.
Never use the container's nominal frame rate to infer correspondence.

The edit spans `[floor(N/3), floor(2N/3))` on that measured source clock. Require
original context before and after it. Prepare the same six cases for every
group, without response-dependent selection:

1. Unchanged original pictures.
2. First publisher swap's native pictures inserted only inside the interval.
3. Second publisher swap inserted in the same way.
4. Ordinary light change: multiply original RGB by 1.08, round/clip to uint8,
   only inside that interval.
5. Ordinary full-picture Gaussian blur at sigma 2 native pixels inside it.
6. Ordinary temporal step: use the original picture two source indices ahead
   inside it, then restore the exact original picture correspondence afterwards.

The three ordinary edits remain non-AI despite measurable changes. Publisher
swap provenance labels the inserted source frames; RGB differences can include
ordinary pipeline/compression effects. Actual changed-frame support is retained,
but it is **not an exact AI-pixel mask or independently verified face identity**.
Record zero-change conditions explicitly, without manufacturing a changed frame.

## Encoding, verification and exhibits

Encode every condition with the same lossless RGB H.264 MP4 recipe: libx264rgb,
QP 0, veryfast, no B-frames, one thread and explicit frame-timing passthrough.
Set the encoder/video track time base
from the denominator of the exact observed frame step. Start the new clock at
zero, preserving each source-relative rational presentation time. Container
support is separate from browser codec playback; detector input is verified
through the app's native decoder, not a browser-decoded preview.

Stream source pictures and retain only a two-frame lookahead for the ordinary
time-step operation. Recheck original source bytes and every consumed native RGB
hash against v92. After encoding, decode every output picture again. Require
byte-identical intended RGB, exact rational PTS, complete constructed display
duration, full frame count and no change
outside the declared interval. Capture before/during/after native PNGs at indices
`a-1`, `floor((a+b-1)/2)` and `b`, bound to their actual output timestamps.
These are construction exhibits, not suspicious regions found by a detector.

Hash-order groups with a fixed v95 salt. The engineering pilot is the first
compatible fitting-role group; reserved model-response roles are not selected
for engineering inspection. The full plan contains 90 groups and 540 cases.
Run each group sequentially under its own exclusive output/profile directory.
Keep every failed/partial attempt and original source; no overwrite, implicit
resume, retry, model fallback or silent source substitution.

Data/decoder/encoder stages use 900 seconds, 1.5-GiB sampled descendant RSS,
3-GiB initial RAM/disk and 1.5-GiB running RAM/disk reserves on project/archive
volumes. Decoder passes have 180-second timeouts; each encoder has 360 seconds.
Resource samples are not guaranteed peaks. No neural library, pretrained model
or generator is loaded, and all existing neural-study limits stay unchanged.

Freeze source/qualified-plan hashes, code/tests, this method, NumPy/OpenCV/psutil
versions and ffmpeg/ffprobe binaries before research rendering. Keep MP4s, PNGs,
per-source fingerprints and individual frame records private on the approved
archive; public results contain aggregates and whole-artifact hashes only.
Scoring, learned-method calibration, lip/audio source separation, additional
mouth generation and customer evidence/export/payment qualification remain
separate unfinished work. No outreach, Actions run, real charge or GCP action
is part of construction.
