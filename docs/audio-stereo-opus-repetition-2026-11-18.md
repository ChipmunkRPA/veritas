# Standalone stereo repeats after Ogg Opus encoding

The existing short stereo repeat check retained 10 of 24 AAC locations
when the same recordings were delivered as 160 kbps Ogg Opus. Opus added
one different correct location and produced no ordinary alerts. This
profile fell below the frozen carrier-retention requirement. The stereo
Opus repeat cue stays in evaluation.

## Paired test

All 256 exact PCM parents from the earlier short stereo AAC holdout
were encoded as constant 160 kbps Ogg Opus. The paired comparator is the
retained 160 kbps AAC reading. The same lower-band stereo procedure
reduces foreground shared between the channels and looks for repeated
quieter sound. It reads one submitted recording at a time. The previous
study used 32 source groups in four DEMAND environments and real speech
from two AMI meetings. Later encodings replay those exact groups and edit
placements. These are staged background repeats with prior project
exposure. A useful location needs at least 0.50 intersection over union
for each occurrence. [Frozen plan](evaluation-panels/audio-stereo-opus-repetition-plan-2026-11-18.json)
· [Prior native held controls](audio-aac-repetition-2026-11-17.md#fresh-local-comparison)

| Comparison outcome | 160 kbps AAC | 160 kbps Ogg Opus |
| --- | ---: | ---: |
| Repeat locations, speech 10 dB louder | 19/32 | 8/32 |
| Repeat locations, speech 18 dB louder | 5/32 | 3/32 |
| Edited files with a pair but no correct location | 6/64 | 4/64 |
| Edited files assessed | 62/64 | 64/64 |
| Ordinary files assessed | 110/192 | 118/192 |
| Ordinary files with a reported pair | 0/192 | 0/192 |

At 10 dB louder speech, seven AAC locations were kept and one different
location was added. At 18 dB louder speech, three were kept. The useful
Opus locations came from the street and metro backgrounds. There were
zero flags on ordinary recordings, changes in foreground direction,
same-channel delivery, repeated foreground speech, and the other
matched controls. The ordinary denominator includes unassessed files,
so coverage is recorded separately.
[Case measurements](evaluation-panels/audio-stereo-opus-repetition-held-2026-11-18.json.gz)

## Why the locations were lost

The page CRCs, packet inventory, and independently decoded frame clock
matched on all 256 files. Each had 188 packets of 960 samples, a 312-sample
pre-skip, 168 trailing samples discarded, and 180,000 presented samples at
48 kHz. Playback began at zero. The foreground alignment also recovered
the exact authored inter-channel delay on all 64 edited files. On the 14
missed AAC locations, Opus produced no reported pair in 12 and produced
only misplaced pairs in two. [Ogg Opus timing specification](https://www.rfc-editor.org/rfc/rfc7845.html#section-4.3)

A diagnostic reran those 14 files with the exact interval lag supplied
from the labels, while preserving the ordinary correlation requirement.
The unmodified search proposed the correct lag on seven cases without
localizing them. Supplying the lag recovered four of the other seven
cases. The remaining ten still lacked a useful range with that known
lag. Search nomination accounted for some losses, and the measured band
support or range selection was insufficient on most. The diagnostic uses
labels after the held comparison and does not enter app inference.
[Known-lag results](evaluation-panels/audio-stereo-opus-repetition-lag-diagnostic-2026-11-18.json)

## Replay

The DEMAND recordings are CC BY-SA 3.0 and the AMI recordings are
CC BY 4.0. The receipts retain source and encoded SHA-256, decoded native
clock, and measurements. The original and rendered audio stays local.
Restore `held-wav` with the earlier AAC preparation instructions. Native
replay uses FFmpeg 7.0.2 with libopus. The recount needs only Python's
standard library.
[DEMAND archive](https://zenodo.org/records/1227121)
· [AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
· [Exact original recipe](audio-aac-repetition-2026-11-17.md#application-and-independent-replay)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
module=scripts.audio_stereo_opus_repetition
held=docs/evaluation-panels/audio-stereo-opus-repetition-held-2026-11-18.json.gz
python3 -B -S -m "$module" review "$held"
backend/.venv/bin/python -B -m "$module" measure "$new_receipt" "$held_wav"
backend/.venv/bin/python -B -m "$module" diagnose "$held" "$held_wav"
```
