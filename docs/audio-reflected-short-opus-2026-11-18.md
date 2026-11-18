# Reflection for 750 ms speech localization in noisy Opus

The earlier Opus comparison found that a two-second range cannot meet
the localization criterion for a splice shorter than 600 ms. This
study examines whether scoring the same original 750 ms samples as
either their direct waveform or that waveform followed by its reversal
helps recover more of these edits. The output range always follows the
original 16 kHz sample clock. Repeating a reflected waveform adds
model context while keeping the original 750 ms evidence width.
Prior research on partial speech deepfakes trains with reflected
segments of different lengths. Here the transform is tested with the
already installed detector at inference. [Earlier Opus comparison](audio-two-second-opus-transfer-2026-11-18.md) · [Reflection and partial speech localization research](https://arxiv.org/html/2604.02913v1#S2.SS2) · [Model terms](commercial-model-audit.md)

The frozen scan advances 500 ms at a time and requires speech activity
for at least half of the window. Each retained range needs another
positive range one grid step away. Each waveform view has its own
operating point, set to the highest lower response of any adjacent pair
on the eight ordinary or isolated-room calibration deliveries. Held
cases use fresh -10 dB remixes of retained signed foreground audio and
native AMI room audio. The 18 held deliveries include eight edited
utterances across five generation families, eight genuine utterances
including the exact originals of three edited utterances, and two
isolated room deliveries. Reader IDs, room IDs and AMI participants
are disjoint across calibration, held and the prior Opus comparison.
The underlying archived utterances and meetings had been used in other
local studies. All deliveries use the prior comparison’s exact 64 kb/s
constant-rate Opus settings. [LlamaPartialSpoof](https://zenodo.org/records/14214149) · [LibriSpeech](https://www.openslr.org/12/) · [LibriTTS](https://www.openslr.org/60/) · [AMI terms and attribution](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

A useful range covers at least half of one annotated edit, with temporal
intersection over union at least 0.30. Matching is one to one. Both
short views are measured as independent supplements to the same
three-second baseline and the earlier strict two-second Opus comparator.
The baseline, comparator and new short views all use the same decoded
Opus recording for each test case.

## Local results

| Evidence | Edited files flagged | Edits located | Edits under 600 ms located | Genuine files flagged | Isolated rooms flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three-second baseline | 2/8 | 2/55 | 0/27 | 3/8 | 0/2 |
| Baseline and strict two-second Opus comparator | 2/8 | 2/55 | 0/27 | 4/8 | 0/2 |
| Comparator with direct 750 ms ranges | 3/8 | 3/55 | 1/27 | 4/8 | 0/2 |
| Comparator with reflected 750 ms ranges | 3/8 | 6/55 | 2/27 | 5/8 | 1/2 |

The 750 ms grid could locate 25 of the 27 short edits if every grid
position were usable and correctly scored. The fixed two-second
width cannot meet the matching rule for any of those 27 edits. Both
short scans assessed 151 calibration windows and 352 held windows.
Median scoring time per held file was 6.65 seconds for the direct
pass and 8.14 seconds for the reflected pass, measured on the same
machine using two inference threads. The three- and two-second
comparators together took a median of 9.63 seconds per file.

The direct scan adds one correctly located edit. The **2.500–3.250 s**
range overlaps the annotated **2.340–2.850 s** edit. It also adds
ranges to three genuine files that already carried a comparator
flag. The reflected scan adds four locations across two edited
utterances and keeps the comparator's two correct locations. For
example, **1.500–2.250 s** overlaps a **1.738–2.168 s** edit. The
same scan also flags **6.500–8.250 s** through three neighboring
windows in the matched genuine recording, as well as two windows
in one isolated room recording. The newly flagged genuine recording
is the exact original of the edited utterance that supplied three
of the four new locations. All added useful locations across both
short scans come from mixes using the same held meeting background.

The shorter windows improve localization and reveal additional false
signals under strong competing speech. Both short
checks remain evaluation candidates. The smaller output window
alone gives no basis for increasing a file's confidence in an
authenticity finding.

## Replay

The [public receipt](evaluation-panels/audio-reflected-short-opus-2026-11-18.json.gz)
contains the source commitments, numeric responses, range positions and
audit counts. The standard-library reviewer recalculates both points
and all outcomes. Independent re-encoding and FFprobe reproduce the
Opus clocks, file hashes and decoded samples for all 26 deliveries.
A separate freeze run reproduced the plan and all signed PCM files.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.review_audio_reflected_short_opus verify \
  docs/evaluation-panels/audio-reflected-short-opus-2026-11-18.json.gz
```

For native replay, retain the three signed source archives and the
AMI WAV paths listed in the frozen plan, together with the official
annotation ZIP. Use an empty output directory and the pinned weights
from the earlier comparison.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
runner=backend/.venv/bin/python
prior=docs/evaluation-panels/audio-two-second-opus-transfer-2026-11-18.json.gz
"$runner" -B -m scripts.audio_reflected_short_opus freeze \
  "$source_archive" "$room_archive" "$ami_annotations" "$prior" "$out"
"$runner" -B -m scripts.audio_reflected_short_opus score \
  "$out" "$out/calibration.json.gz" calibration
"$runner" -B -m scripts.audio_reflected_short_opus score \
  "$out" "$out/held.json.gz" held --calibration "$out/calibration.json.gz"
python3 -B -S -m scripts.review_audio_reflected_short_opus build \
  "$out" "$out/verified.json.gz"
```
