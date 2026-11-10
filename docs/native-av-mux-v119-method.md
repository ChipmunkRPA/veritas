# Native audiovisual assembly qualification

Before generating complete native mouth-edit clips, qualify a lossless container
path independently of the neural renderer. Keep the successful v118 renderer,
source roles and all previous outputs unchanged. This stage performs no inference
and inspects no detector response.

Use isolated publisher-pinned PyAV to encode Matroska FFV1/bgr0 video and mono 16-kHz
signed-16 PCM. Preserve every supplied picture's integer millisecond PTS and
every audio sample, including the nonnegative audio origin. The input contract
requires 2–200 strictly increasing nonnegative picture timestamps and an audio
origin on the millisecond lattice, matching this CREMA-D panel. Reject unsupported
origins instead of rounding/rebasing them. Keep variable picture steps and gaps.
Use one encoder thread; no external model, service or shell command.

The application environment has no PyAV. An initial preparation identity read
stopped before creating a trial or media stage; preserve that source snapshot
and failure record. Acquire the official compatible av19.0.2 wheel into a
separate research overlay using interpreter compatibility tags. Verify its
publisher size/SHA-256, install without dependencies/compilation into that overlay,
then bind installed RECORD hashes, package version and bundled codec versions.
The [pinned runtime plan](evaluation-panels/native-av-runtime-v119.json) records
the publisher's BSD-3-Clause package declaration. The application environment is
unchanged. [PyAV's installation documentation](https://pyav.basswood.io/docs/stable/overview/installation.html)
describes the bundled FFmpeg wheels; package and bundled library identities are
separate from claims of independent decoder implementation.

Set each video packet's duration from the next observed picture timestamp;
construct only the terminal packet's duration from the upper median positive
step. This defines a shared output display tail, not an inferred source exposure
or proof of the original file's final duration. The packet input time base is
1/1000; PCM frames have 1/16000 time base and fixed 1024-sample chunks plus the
complete tail. Container nominal rate metadata is not a resampling instruction.
Require exact clock/bytes after decoding; metadata alone is insufficient.

Run exactly two authored cases and one fixed fitting-source rewrite. The first
authored colour pattern has jittered timestamps and an explicit 100-ms picture
gap, with audio at zero. The second has video at 67 ms and audio at 42 ms,
including a partial final audio packet. Use deterministic integer PCM and RGB
values. Then reassemble the complete first v116 fitting source, using all its
original native frames and the complete qualified v105 mono16k waveform. Rehash
each source and verify each input RGB against v92 before encoding. No selection
or confirmation source is consumed by this qualification.

For each output use the application command decoder/probe to compare every RGB
hash, integer PTS, time base, frame count and dimensions. Use the application
audio decoder and sample-source mapping to require complete exact signed-16
sample bytes/count, unchanged audio origin and one continuous sample mapping.
PyAV and command FFmpeg share codec lineage; this is a second API path, not an
independent codec implementation or external human review.

Freeze core/tests/procedure, input/renderer commitments and package versions
before running. Reuse the v108 resource runner's exact code object in a new
exclusive context, `bind` data policy and fixed worker. Upstream verification
requires the qualified owned-descendant monitor. Keep 900 seconds, 1.5-GiB
sampled descendant RSS, 3-GiB initial project/archive disk and RAM, and 1.5-GiB
running reserves. Execute outside the sandbox; periodic samples are not peaks.
Preserve partial files and profiles on any failure; no retry, overwrite,
fallback, point/role change or relaxed resource requirement.

Store output media and per-source frame/sample fingerprints privately on Seagate.
Publish code, aggregate counts and whole-result commitments only. This qualifies
assembly inputs for subsequent known-boundary mouth generation, not achieved
phonetic synchronization, deepfake detection, calibration or serving confidence.
No public raw media, external contact, Actions, real charge or GCP deployment.
