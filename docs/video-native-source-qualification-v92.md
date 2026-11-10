# Shared video sources: native clocks and decoded aliases

This data-only qualification precedes new detector fitting or comparison. It
uses the exact 336 assets in the [digest-verified acquisition](video-common-source-acquisition-v91.md).
No generator, detector, remote Python, neural weights or customer inputs are
loaded. The initial freeze binds source and historical journals, code/tests,
this method, NumPy/psutil versions and ffmpeg/ffprobe binaries. Freeze and
aggregate results may be public; individual source records remain private.

## Measurements

- Decode every video frame at native display resolution using the app's PTS
  reader. Store the complete native time base, integer presentation timestamps,
  dimensions and RGB hashes. Compare actual delivered indices/times and frame
  count with the table. Do not substitute nominal frame rate. Truncation,
  interpolated timestamps or non-increasing PTS prevent clock qualification.
- Inspect audio streams explicitly. For the first audio stream, retain decoded
  frame timestamps and sample counts on its own rational clock. Record missing
  support, gaps and overlaps. A gap greater than one stream tick plus one audio
  sample prevents a contiguous label; unresolved overlap/missing clocks prevent
  audio-clock qualification. Audio presence alone does not establish speech.
- Decode canonical signed-16 mono 16-kHz audio for content comparison only.
  This resampling does not replace native audio timing or become synthesis
  evidence. Keep its sample count, full content digest and informative one-second
  block sequences. A block is informative when its sample range is at least 64,
  standard deviation at least 32 and it has at least 64 distinct signed-16 values.

## Source connections and roles

Join planned groups sharing an exact complete RGB sequence with informative
sequence support, three consecutive distinct nonuniform RGB frame hashes, a
complete informative canonical PCM stream, or three consecutive distinct
informative PCM blocks. Uniform/static video and silent audio are insufficient
to connect otherwise distinct groups. These are conservative content links,
not biometric identity claims. Use every decoded frame, not only detector samples.

Compare new RGB triples with the complete saved v18 RGB audit journal. Exclude
the entire connected component on any historical match or failed/unqualified
video member. Preserve all excluded records and reasons. Historical audio and
other old panels without recorded RGB sequences are not independently compared
by this step; the prior metadata exclusions remain in force.

If a refined component crosses provisional roles, reassign all components in
fixed salted-hash order round-robin across fitting, confidence calibration,
selection and confirmation. Otherwise retain the provisional roles. No detector
response is used. Exact links do not establish independence from arbitrary
crops, re-encodes, near duplicates, unknown shared people or pretraining data.

## Execution and limits

Run `scripts.video_native_runtime_v92 prepare`, publish the freeze, then
`qualify 0` through `qualify 11` sequentially (28 assets per batch), followed
by `finalise`. Every journal, profile and result is exclusive. Preserve failed
or partial attempts; no overwrite, implicit resume or automatic retry. A failed
asset remains explicit while other planned assets continue. Failed/guarded
batch runtimes cannot qualify. No failed source becomes an original verdict.

Each parent samples its worker and decoder descendants: 900-second stage bound,
1.5-GiB sampled tree RSS cap, 3-GiB initial available RAM and free disk, and
1.5-GiB running RAM/disk reserves on internal and archive volumes. Per-operation
decoder/probe timeouts are 120 seconds. Process-group cleanup applies on guard
or failure. Samples are lower bounds on peaks, not continuous guarantees.
These data/decoder-only limits do not change any neural experiment's larger
memory requirement or scientific settings.

Authored tests include a real lossless decoder round trip with irregular frame
timestamps, audio and silent variants; rational sample clocks with gaps,
overlaps and missing support; source-component propagation; historical matches;
static/silent non-links; byte-plan tampering and refusal to overwrite. These
validate qualification behavior, not detector accuracy.

The output is still **not a detector-ready benchmark**. Known edit boundaries,
native correspondence, ordinary controls, new mouth manipulations, each model's
commercial grants and the fixed shared scoring/calibration protocol remain
separate requirements. No outreach, Actions run, real charge or GCP deployment
is part of this qualification.
