# Time-local lag proposals for quieter repeated sound

The stereo Opus trial identified some missed repeats whose true offsets
were absent from the whole-file proposals. This study compares a second
way to choose offsets using a single submitted recording. It uses neither
another copy of the recording nor a learned model.
[Earlier known-lag diagnostic](audio-stereo-opus-repetition-2026-11-18.md#why-the-locations-were-lost)

The candidate reads 120 ms pieces of the same quieter residual at 80 ms
steps. It looks for another part of that residual where the waveform
correlation in two frequency bands reaches 0.62. Up to two local peaks
from each piece enter a vote. The 12 best coarse offsets and their nearby
fine positions go through the existing native-clock interval verification.
The existing result has priority. The local candidate runs only when an
assessed file returned no pair. Its thresholds are fixed and carry zero
AI-confidence weight. [Frozen plan](evaluation-panels/audio-stereo-local-lag-plan-2026-11-18.json)

## Source roles

The development run uses eight groups from two DEMAND rooms with real AMI
speech from two sessions. It measures 16 staged background repeats and
48 ordinary controls. The existing Opus procedure locates one repeat.
The local proposal method raises this to three with zero new ordinary
or misplaced alerts. No development score was used to tune a threshold.
The exact measured records were frozen before opening the held comparison.
[Development receipt](evaluation-panels/audio-stereo-local-lag-calibration-2026-11-18.json.gz)

The held run uses 24 groups from separate DEMAND kitchen, field, street
and metro recordings with foreground speech from two separate AMI
sessions. Its primary excerpts are separate from those in the earlier
short AAC and playback-speed comparisons. One of the six replacement
control excerpts per room overlaps the earlier playback-speed study.
Source recordings have prior project exposure.
The held set stages two repeated background versions per group with
speech 10 or 18 dB louder. It also uses six matched ordinary controls
per group. Repeats retain both exact source intervals.

FFmpeg encodes each held stereo parent at constant 160 kbps in Ogg Opus
and at nominal 160 kbps in AAC. The decoded presentation starts at zero
and contains exactly 180,000 stereo samples in both formats. Opus page
and packet checks verify pre-skip and end trim. The independently inspected
stereo decoded frame clock agrees. The matcher requires 0.50 temporal
intersection over union on each of the two occurrences. The comparison
measures audio edits staged on authentic sound and real speech.
[DEMAND recordings and CC BY-SA 3.0 license](https://zenodo.org/records/1227121)
· [AMI recordings and CC BY 4.0 license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Held results

| Check | Located, speech 10 dB louder | Located, speech 18 dB louder | Misplaced alerts on edited files | Ordinary alerts | Ordinary files assessed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched AAC | 13/24 | 5/24 | 7/48 | 6/144 | 84/144 |
| Opus whole-file lag search | 8/24 | 2/24 | 5/48 | 3/144 | 86/144 |
| Opus with local lag fallback | 9/24 | 3/24 | 6/48 | 6/144 | 86/144 |

All 48 edited files were assessed in each arm. The new exact pairs came
from different street and metro excerpts, one at each loudness. They
retained the earlier ten useful Opus locations and raised the number of
matched AAC successes kept from ten to 11 of 18. The additional incorrect
pair and all three new ordinary alerts came from the same metro excerpt.
Those four files shared closely matching ranges near 2.15 and 3.37 s,
including the ordinary panned clip and the foreground-repeat control.
The two correct additions had five and ten tested fine lags. Each of the
four incorrect additions had 60. This observation was made after held
scoring and would need a new prospective test before adding a limit.
The candidate remains experimental because the frozen improvement and
adoption gates both failed. The published observations include all
corresponding AAC and Opus input hashes and native ranges.
[Held measurements](evaluation-panels/audio-stereo-local-lag-held-2026-11-18.json.gz)

## Reproduce locally

Use the source rosters and preparation locations from the
[native stereo test](audio-native-alignment-2026-11-17.md#app-and-independent-replay)
and the [AAC comparison](audio-aac-repetition-2026-11-17.md#application-and-independent-replay).
The complete `ES2011b.Array1-05.wav` and `IS1006b.Array1-05.wav` files come
from the [stereo timing input](audio-stereo-timing-change-2026-11-17.md).
The native test uses FFmpeg 7.0.2 with libopus. The runner verifies every
source file's hash before reading it and keeps
only one case's transcodes in its temporary work directory. The numeric
review uses Python's standard library and the retained observations.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
module=scripts.audio_stereo_local_lag
base=docs/evaluation-panels/audio-stereo-local-lag
cal="$base-calibration-2026-11-18.json.gz"
held="$base-held-2026-11-18.json.gz"
python3 -B -S -m "$module" review "$cal"
python3 -B -S -m "$module" review "$held" --calibration "$cal"
backend/.venv/bin/python -B -m "$module" calibration "$new_cal" "$dev_rooms" "$old_ami"
backend/.venv/bin/python -B -m "$module" held "$new_held" "$held_rooms" "$other_ami" --calibration "$new_cal"
```
