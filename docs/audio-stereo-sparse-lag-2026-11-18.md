# Sparse local lag search for quieter stereo repeats

This test asks whether restricting a fallback to well-supported time offsets
improves the location of repeated background sound under speech after Ogg
Opus encoding. It uses only the submitted stereo recording. The measured
cue concerns repeated sound and carries zero AI-confidence weight.

## Frozen comparison

The previous local search gained two correct locations and added four
incorrect highlights in one metro excerpt. The prior run observed fewer
tested offsets on the correct additions. This new comparison keeps the
same coarse cross-band search, then rechecks each neighboring fine offset
at 16 kHz against the anchor that proposed it. At least two frequency
bands must have the original minimum energy and correlation of 0.62.
The fallback abstains if more than 12 distinct fine offsets qualify.
It runs after an assessed whole-file search returns no interval. The
existing range verifier still has to find both locations.
[Previous local lag study](audio-stereo-local-lag-2026-11-18.md)
· [Frozen plan](evaluation-panels/audio-stereo-sparse-lag-plan-2026-11-18.json)

The 24 primary groups use distinct 48 kHz excerpts from four recorded
backgrounds (metro, station, street and restaurant) and native 16 kHz
foreground clips from four AMI meetings. Selected AMI clips start after
the earlier 600-second room sampling range. The room and meeting
recordings have prior project exposure. The panel is a prospective test on
new excerpts for this check. An audit before reading the results found two
0.64-second source intersections. The TMETRO-122 replacement intersects an
earlier narrowing-study excerpt. The STRAFFIC-33 main block intersects an
earlier AAC replacement excerpt. The report includes a sensitivity count
with both groups excluded and the numerical acceptance limits held fixed.

Each group has a staged repeat with foreground speech nominally 10 or
18 dB louder, plus seven controls without an authored repeated background.
These include an ordinary spoken mix, a different background excerpt,
moving foreground speech, repeated foreground speech, identical channels,
the unedited room and a motor-like sweep that changes its rate. The held
set also includes two new deliveries from each of 30 separate DCASE
bus, subway train and subway station recordings. A 3.75-second continuous
excerpt at second 17 and an ordinary foreground overlay at second 24
challenge the check with real acoustic variation. The full DCASE recordings
also appeared in an earlier study of stereo narrowing. The DCASE excerpts
originate at 44.1 kHz and are converted to 48 kHz before encoding.

- [DEMAND source recordings and CC BY-SA 3.0 license](https://zenodo.org/records/1227121)
- [AMI Meeting Corpus and CC BY 4.0 license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
- DCASE 2013 public scene recordings by D. Stowell and the DCASE organizers. [Publisher archive](https://archive.org/details/dcase2013_scene_classification) · [CC BY 2.0 UK license and dataset citation](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/dcase2013_public.html)

Both 160 kb/s encodings use the identical stereo PCM parent for each case.
AAC and Opus both present 180,000 samples at the original presentation
rate. Opus packet counts and timing independently agree with the decoded
48 kHz sample clock. An edit is located only if both returned intervals
overlap at least half the union of their corresponding source labels.

## Held results

| Outcome | Matched AAC | Opus whole-file search | Opus with sparse fallback |
| --- | ---: | ---: | ---: |
| Located with speech 10 dB louder | 21/24 | 15/24 | 19/24 |
| Located with speech 18 dB louder | 14/24 | 5/24 | 5/24 |
| Wrong location on edited files | 2/48 | 6/48 | 6/48 |
| Control files with a reported pair | 5/228 | 3/228 | 3/228 |
| Control files assessed | 143/228 | 144/228 | 144/228 |
| Real transit controls with a reported pair | 0/60 | 0/60 | 0/60 |

All 48 edited files were assessed. The fallback added four correct pairs
from three environments, each with five surviving fine offsets. Three
were recovered AAC locations. The fourth was a different useful Opus
location. All reported pairs on controls came from the crafted motor
sweeps. The fallback added zero alerts on controls and zero misplaced
edit locations. Of 163 fallback opportunities, eight had more than
12 surviving positions and 134 had none.

| Newly located source excerpt | First occurrence, seconds | Second occurrence, seconds |
| --- | ---: | ---: |
| Metro, room excerpt at 224 s | 0.440–0.680 | 2.357–2.597 |
| Station, room excerpt at 17 s | 0.400–0.650 | 2.208–2.458 |
| Station, room excerpt at 109 s | 0.440–0.690 | 2.357–2.607 |
| Restaurant, room excerpt at 15 s | 0.490–0.640 | 2.298–2.448 |

The whole-file Opus check kept 20 of the 35 correct matched AAC locations.
The sparse fallback retained 23 of 35 (65.7%). It met every experimental
improvement condition and fell below the 85% AAC retention required for
release. The method remains in evaluation. Excluding the two source
groups with earlier intersections retained all four gains (19 to 23 of
44), with one existing control alert on the 214 remaining controls and
zero new alerts. AAC retention in that subset was 22 of 32 (68.8%).
[Held case measurements](evaluation-panels/audio-stereo-sparse-lag-held-2026-11-18.json.gz)

## Replay locally

Prepare `native/` and `aac/` with the matching publisher channel pairs
from the [native](evaluation-panels/audio-native-alignment-sources-2026-11-17.json)
and [AAC](evaluation-panels/audio-aac-repetition-sources-2026-11-17.json)
source rosters. Put the four AMI `<meeting>.Array1-05.wav` tracks and
publisher `scenes_stereo.zip` in the parent directory. The runner
verifies the exact source digests. Source audio and transcodes remain
local. Use FFmpeg 7.0.2 with libopus to measure again.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
module=scripts.audio_stereo_sparse_lag
receipt=docs/evaluation-panels/audio-stereo-sparse-lag-held-2026-11-18.json.gz
python3 -B -S -m "$module" review "$receipt"
backend/.venv/bin/python -B -m "$module" measure "$run/held.json.gz" "$sources"
```
