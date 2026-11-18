# Frequency-dependent stereo foreground cancellation

The existing stereo repeat check aligns two channels and subtracts a
fixed share of the common foreground before looking for repeated
quieter sound. This trial instead estimates a complex gain in each
well-supported frequency bin. It tests whether disagreements left by
Opus encoding can be reduced while preserving a repeated background
and its original sample clock.
[Earlier stereo Opus results](audio-stereo-opus-repetition-2026-11-18.md)
· [Research on spatial and spectral filtering](https://arxiv.org/abs/2509.25982)

The filter uses the existing native alignment and the upper 35% of
frames by energy to fit one response for the full
recording. It bounds adjustments to the fixed gain and uses the
existing correlation and range selection on the residual. No supervised
fit, reference audio, or neural weights enter the proposal. A useful
pair requires at least 0.50 intersection over union for **each** occurrence
on the original 48 kHz time base.

## Local comparison

The frozen candidate measured 64 development deliveries and another 96
deliveries from separate room and foreground recordings. Each group
contributes an unmodified room, ordinary and moving speech mixes, a
background replacement, repeated foreground speech, identical channels,
and staged repeats with speech 10 or 18 dB louder than background.
These are the same encoded mixtures from a separate envelope diagnostic.
The underlying public recordings had also appeared in earlier work.
The results measure the added processing on those files.
[DEMAND terms and attribution](https://zenodo.org/records/1227121)
· [AMI terms and attribution](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

| Outcome | Current comparison | With frequency-dependent fallback |
| --- | ---: | ---: |
| Development edit locations | 1/16 | 1/16 |
| Development controls flagged | 0/48 | 0/48 |
| Transfer edit locations, speech 10 dB louder | 4/12 | 4/12 |
| Transfer edit locations, speech 18 dB louder | 3/12 | 3/12 |
| Transfer controls flagged | 0/72 | 0/72 |

The alternative alone located the same seven transfer edits. Its endpoint
selection shifted on three edited metro deliveries. It introduced no
useful fallback locations and remains outside application scoring.
All 160 file hashes and decoded sample hashes matched the earlier
native reads. The timing check reproduced the 48 kHz presentation for
each file. [Numeric receipt](evaluation-panels/audio-stereo-complex-projection-2026-11-18.json.gz)

## Replay

The numeric replay recalculates the count of correctly paired ranges,
ordinary flags and misplaced additions. The receipt contains only
source references, file hashes, timing, paired ranges, and scalar
measurements.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_stereo_complex_projection review \
  docs/evaluation-panels/audio-stereo-complex-projection-2026-11-18.json.gz
```

The native commands take the original channel and voice tracks. Use
the `native` and `aac` DEMAND directories from the [stereo source instructions](audio-stereo-sparse-lag-2026-11-18.md#replay-locally).
Add the publisher tracks `ES2004a.Array1-05.wav` and
`TS3004a.Array1-05.wav` to the AAC directory. The `voices` directory
contains the four AMI tracks from that earlier study. The replay hashes
all source bytes before measurement, then checks each rendered Opus
file and its presentation clock.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.audio_stereo_complex_projection
"$runner" -B -m "$module" freeze "$native" "$aac" "$voices" "$plan"
"$runner" -B -m "$module" measure "$native" "$aac" "$voices" "$plan" "$output"
```
