# Native audio and conditioning waveform qualification

This data-only continuation uses all 48 sources and the immutable
[sentence/time-support plan](av-sentence-support-v104-result.md). It prepares
private native and mono-16-kHz WAV caches with source-time coordinates. No
generator, speech encoder or detector is executed. Source roles, old audio
calibration and serving behavior are unchanged.

## Before source decoding

Freeze this implementation, authored tests, procedure, source/role/clock plans,
FFmpeg binary and monitoring implementation. Publish and verify the freeze
before decoding these sources. Check authored mono/stereo PCM fixtures first:
two seconds at 44.1 kHz with impulses at five fixed native positions including
the beginning/end. Require 32,000 mono output samples and a nonzero peak within
one output sample of each exact rational expected position. This tests the
pinned resampler on authored impulses, not every waveform or original codec's
capture delay. Channel assignments alternate in the stereo fixture.

## Native PCM and source commitments

For every source, verify acquired file SHA-256 and preserve its recorded native
rate and channels. Decode signed-16 little-endian PCM with no requested channel
or sample-rate change. Require its sample-frame count to equal the sum of all
native audio-frame counts in the v92 record and v104 plan. Independently invoke
FFmpeg's hash muxer on a second source decode, requiring the same PCM SHA-256.
Both use the same decoder library; this is an exact repeat, not an independent
codec engine or source-origin proof.

Wrap those native samples in WAV without changing the payload, then reopen and
verify channels, sample width, sample rate, frame count and every byte. Preserve
stereo in the native cache. Decode the separate mono-16-kHz conditioning input
with the existing source conversion recipe and require exact equality to both
the earlier v92 canonical sample count and PCM hash. Cache that WAV separately.
The mono input does not replace the original channels for other analyses.

## Frame/sample correspondence

For every common native picture center, calculate the exact rational coordinate
`(source_picture_time - source_audio_origin) * 16000`. Retain its floor/ceiling
sample indices, declared source time, source video index and native clock
quantization allowance. The original audio lead is preserved. Do not treat a
video-relative zero as audio sample zero. Out-of-range sample brackets remain
unavailable rather than being padded, clamped or called genuine.

The native timestamp's one-clock-tick-plus-one-sample allowance remains
explicit. The impulse allowance does not remove that uncertainty, prove captured
phonetic synchronization, or qualify a generator's encoder stride, feature
centers or model padding. `conditioning_waveform_ready` means source bytes,
sample counts, conversion commitments and picture/sample bounds passed with
nonzero mono samples. `generator_conditioning_qualified` stays false. A short
clip may have a valid waveform while lacking a supported 20-native-frame edit.

## Resources, failures and publication

Use the existing data-only monitor: 900 seconds and 1.5 GiB sampled parent plus
active-command descendant RSS, 3 GiB initial available RAM/disk and 1.5 GiB
running RAM/disk reserves. Each FFmpeg command times out after 180 seconds.
Inspect permissions on an owned standard-library child before launch outside
the sandbox. No neural budgets are relaxed. Samples are not guaranteed peaks.

Output/profile/cache paths are exclusive. Preserve every failed or partial
attempt, with no automatic retry, overwrite, source substitution or fallback
resampler. Keep WAVs, source identities, individual fingerprints and frame/sample
coordinates private on the approved Seagate archive. Publish code, aggregate
results and whole-artifact hashes after actual terminal/profile inspection.
No raw source media, customer job, model download, outreach, GitHub Actions,
real charge or GCP action is part of this preparation.
