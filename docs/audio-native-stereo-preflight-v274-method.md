# Native stereo audio preflight

The current audio and video paths score the first audio stream after FFmpeg
downmixes it to mono. This preflight checks exact observed channel recovery
on authored native 16 kHz PCM and verifies archived two-microphone background
inputs for a future comparison. It opens no foreground audio and performs no
model inference, calibration, product changes or media publication.

## Archive and future role plan

Reuse the verified DEMAND washroom and river archives in development and the
kitchen and traffic archives in holdout. Verify the original publisher MD5,
the archive SHA-256 and size against v204 and v212, and the archive rights and
attribution. Each selected member, ch02 and ch16, must be unique within its
ZIP, native signed-16-bit mono at 16 kHz, and exactly 4,800,064 samples. Read
its complete PCM after checking the header, require distinct channel hashes,
and bind all planned excerpts in order. This reads the archives without
extracting or writing media. The four parent recordings have previous project
exposure on channel one. The recorded microphone pair and mixes for this
study are new project views of those sessions. [DEMAND publisher](https://zenodo.org/records/1227121)

Use 16 three-second excerpts starting at `200 + 6 × ordinal` seconds per
tape. For a later composition stage, assign the exact v169c source manifest
order within each role and origin family, cycling the excerpt ordinal modulo
16. Alternate the lower-interference channel by the same ordinal's parity
for the future controlled asymmetric condition. The v169c role separation
and v172 preparation receipts are bound. The 37 development and 40 holdout
foregrounds are reserved for a separate composition and scoring procedure.
An eventual stereo mix would pair an injected known-origin mono foreground
with synchronized microphone background channels. The construction cannot
establish performance on naturally captured stereo speech.

## Authored FFmpeg and clock checks

Write only short temporary authored 16 kHz stereo fixtures with signed impulses
in both channels, swapped channels, each channel silent in turn, and opposite
phase. Decode each through the existing production entry point. Require exact
signed-16-bit PCM and source-range parity with an explicit FFmpeg mono-resampler
call. Use FFmpeg channel selection independently and require byte identity to
each authored channel and equality to the production sample clock. Require exact
impulse support, unchanged mix after swapping, equivalent one-sided mixes,
and cancellation on the opposite-phase fixture. The comparator shares FFmpeg
with production. It does not prove equivalence across codecs, FFmpeg versions,
source time discontinuities or sample rates. The two executable hashes and the
implementation are captured in the readout.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_preflight_v274
```

The readout goes to standard output. It contains archive/fixture commitments
and the frozen plan without raw audio or foreground records. Keep the mono
production route and historical evidence bindings in place. A future candidate
must choose eligible channels using acoustic information before reading any
authenticity score, preserve channel identity and source clocks in evidence,
and pass the separately frozen paired development and holdout comparison.
