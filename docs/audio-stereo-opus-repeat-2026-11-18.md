# Repeated quieter sound in short stereo Ogg Opus

Chipmunk Verify can inspect continuous 3–4 second stereo Ogg Opus audio
in development with `VERITAS_AUDIO_STEREO_OPUS_REPEAT_REVIEW=true`.
The check verifies a zero-origin 48 kHz playback clock, a single
two-channel stream and 20 ms constant-rate packets at 160 kb/s.
It reduces dominant sound shared by the stereo channels and presents
both likely repeat intervals with original stereo playback, reduced
audio and waveforms. JSON, PDF and the signed offline package use the
same source positions. The result carries zero AI-confidence weight.
Repeated speech, loops and ordinary editing can produce similar sound.

An opted-in Ogg Opus recording outside the supported profile receives
an unavailable coverage message. An eligible recording with too little
shared sound can also be unassessed. A supported search that finds
nothing reports that no qualifying repeat was found in the portion
checked.

## Frozen method and confirmation

The first pass uses the existing whole-file lag nominations and fixed
lower-frequency waveform search. When that pass can assess the file
and returns no pair, a second pass nominates local offsets from 120 ms
anchors. Two frequency bands must support an offset at coarse and fine
positions. At most 12 distinct fine positions enter the original
window verification. This lane measures acoustic repetitions separately
from the speech-origin assessment. [Prior sparse-lag comparison](audio-stereo-sparse-lag-2026-11-18.md) ·
[Research on separating speech and background evidence](https://arxiv.org/abs/2610.14141) ·
[Ogg Opus playback-clock specification](https://www.rfc-editor.org/rfc/rfc7845.html#section-4.3)

The confirmation uses 24 distinct binaural DCASE recordings drawn from
six scenes. Each is combined with a distinct transcript-selected part
of one of four AMI meetings. These AMI participants do not appear in the
earlier stereo repeat development and selection groups. The scene
recordings are also separate from the earlier sparse repeat panel.
The underlying audio sources have prior exposure in other local
acoustic studies. This comparison uses fresh combinations and leaves
the earlier operating rules fixed.

Each group has two staged repeat conditions. The panned competing
speech is nominally 10 or 18 dB louder, and the authored repeat lasts
0.24, 0.40 or 0.64 seconds. Controls include an unchanged binaural
recording, an unedited speech overlay, a background replacement,
moving speech, repeated foreground speech and identical channels.
Eight groups are tested at each of 3.0, 3.5 and 4.0 seconds. Both
reported intervals must have at least 0.50 temporal intersection over
union with their corresponding known occurrences. All 192 files
reproduced the exact source clock, decoded samples and measurements
through the serving runtime. [DCASE recording and CC BY 2.0 UK terms](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/dcase2013_public.html) ·
[AMI CC BY 4.0 terms and attribution](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

| Outcome | Opus whole-file search | With the sparse fallback |
| --- | ---: | ---: |
| Useful pairs, speech 10 dB louder | 21/24 | 24/24 |
| Useful pairs, speech 18 dB louder | 15/24 | 16/24 |
| Misplaced pairs on edited recordings | 3/48 | 3/48 |
| Ordinary recordings with a pair | 0/144 | 0/144 |
| Ordinary recordings assessed | 90/144 | 90/144 |

All four gains came from four-second recordings drawn from a park, an
office and a supermarket. The ordinary count includes 54 files where
the check abstained. The earlier AAC transfer result remains part of
the record. This release is an opt-in Opus check supported by its own
held comparison and ordinary controls. [Earlier carrier comparison](audio-stereo-opus-repetition-2026-11-18.md) ·
[Case-level numeric receipt](evaluation-panels/audio-stereo-opus-repeat-confirmation-2026-11-18.json.gz)

## Local app and replay

The isolated app test runs four USD 9.99 simulated checkouts. It uses
the same sparse-fallback positive with the option off and on, an
eligible ordinary file, and a variable-rate Opus control that the
feature cannot assess. The test compares customer assessments for the
same audio across the off/on pair, matches JSON and signed PDF across
exports, replays the signed packages against the original audio, and
opens both the positive and unsupported cases in a browser.
[App verification receipt](evaluation-panels/audio-stereo-opus-repeat-app-2026-11-18.json)

A separate local replay passed the 52 repeat and stereo regression tests
and all four simulated checkout cases. Chromium played both original stereo
and background-reduced evidence clips. Their waveforms, PDF, JSON and signed
source package agreed on source sample ranges [18,720, 27,360) and
[105,516, 114,156) on the 48 kHz clock. The two settings returned the same
overall assessment for the same original. The ordinary file returned no
repeat and the variable-rate file showed the unavailable coverage message.

The standard-library reviewer reconstructs counts and exact sample
coordinates from the published numbers. It needs no source audio.

```sh
export PYTHONPATH=backend
verify=scripts.review_audio_opus_repeat_confirmation
receipt=docs/evaluation-panels/audio-stereo-opus-repeat-confirmation-2026-11-18.json.gz
python3 -B -S -m "$verify" verify "$receipt"
```

For native replay, supply the pinned DCASE `scenes_stereo.zip`, the AMI
1.6.2 manual annotation ZIP and the pinned AMI files listed in the
source plan. The relative AMI paths start at `$source_archive`.
Set `$out` to an empty working directory and use FFmpeg 7.0.2 with
libopus. Source audio, export packages and browser screenshots stay
in the local working directory.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.audio_opus_repeat_confirmation
"$runner" -B -m "$module" freeze "$dcase_archive" "$ami_annotations" "$source_archive" "$out/freeze.json"
"$runner" -B -m "$module" score "$dcase_archive" "$ami_annotations" "$source_archive" "$out/freeze.json" \
  --out "$out/held.json.gz" --fixtures "$out/app-inputs"
python3 -B -S -m "$verify" verify "$receipt" --native "$out"
"$runner" -B -m scripts.audio_opus_alignment_app "$out/app-inputs" "$out/local-smoke" --browser
```
