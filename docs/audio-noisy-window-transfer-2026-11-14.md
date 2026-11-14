# Can staggered windows or less rumble recover short speech edits?

Two local experiments tested whether the existing voice measurements could
recover short edits under strong background sound. One applied the same window
rules to edits from a different dataset and editing procedure. The other
tested whether removing low-frequency sound could rescue windows that had
already failed the speech gate. Neither added a correct location under noise.

## Staggered windows on the SINE splices

This study reused 15 publisher-labeled SINE cut-and-paste edits and their
matched original recordings, one pair per reader. Labels span about 0.23 to
0.79 seconds within six-second excerpts. Each pair was measured clean and
mixed with rain, train or competing speech at 5 dB above foreground RMS. The
panel also includes 16 background-only controls. The readers were previously
used for one-second refinement, and the recordings differ from the
LlamaPartialSpoof panel used to test staggered windows. The 136 deliveries
reuse those 15 foreground pairs and a smaller number of background recordings.
[SINE inputs and terms](audio-narrow-flag-transfer-2026-11-14.md#new-readers-from-the-publisher-archive)
· [Background pins and terms](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)

The comparator is the existing six-second long-review measurement, which
requires two AI-leaning three-second windows at this duration. The alternatives
add the earlier 750 ms stagger and separately test 2.5-second corroboration.
They keep the original calibration and require at least half the edit to be
covered and an intersection over union of at least 0.30. This comparison does
not measure the separate short-upload review or whole-file accuracy. The
optional earlier-copy check received the clean original and was measured
separately. Its output measures a local drop in matching sound.
[Window schedule](audio-staggered-voice-2026-11-14.md#matched-comparison)
· [Earlier-copy method](audio-reference-variable-window-2026-11-14.md)

| Outcome | Current long review | With staggered 3 s | With 2.5 s | Earlier-copy check |
| --- | ---: | ---: | ---: | ---: |
| Correct locations, clean | 0/15 | 0/15 | 0/15 | 0/15 |
| Correct locations, rain | 0/15 | 0/15 | 0/15 | 0/15 |
| Correct locations, train | 0/15 | 0/15 | 0/15 | 0/15 |
| Correct locations, competing speech | 0/15 | 0/15 | 0/15 | 0/15 |
| Genuine deliveries flagged | 0/60 | 1/60 | 1/60 | 2/60 |
| Background-only controls flagged | 0/16 | 0/16 | 0/16 | 0/16 |

The three-second stagger could have overlapped two labeled edits per scene
under the fixed location rule. Both remained reachable after speech admission
for clean and competing speech, and neither remained reachable for rain or
train. The 2.5-second grid could not meet the location rule on these short
splices, even when every planned window was treated as available. It yielded
three misplaced clean flags and two misplaced rain flags. Only 14 of 90
planned windows on train-masked edits reached voice scoring. The earlier-copy
route admitted the pairs for 36 of 60 edited deliveries, yet located none of
the labeled intervals. Both single-file alternatives missed the frozen target
of three additional useful locations across two noisy scenes and two readers.

## Removing rumble before checking speech activity

The second comparison reused the exact 98 LlamaPartialSpoof deliveries from
the earlier window study. It reapplied the speech gate only to previously
skipped three-second windows, after fixed third-order zero-phase high-pass
filtering at 240 Hz. When a window became eligible, the trial scored its
original samples and its filtered samples separately. The original decision
region was reused as a diagnostic for the filtered audio. A filtered score
has no new calibration here. [Original 98-delivery comparison](audio-staggered-voice-2026-11-14.md#matched-comparison)

| Previously skipped windows | Reached scoring after filtering | Added AI-leaning windows in either arm |
| --- | ---: | ---: |
| Rain-masked partial edits | 0/8 | 0 |
| Train-masked partial edits | 2/66 | 0 |
| Genuine in rain | 2/3 | 0 |
| Genuine in train | 0/41 | 0 |
| Rain-only and train-only | 0/40 | 0 |

No gain-only windows needed reevaluation. Neither scoring arm added a correct
location or a control flag. The limited change at speech admission and the lack
of a scoring gain keep this filter outside the analysis worker. The four new
eligible windows and the source pairs are too few to infer whether a separately
calibrated model for filtered speech would help.

## Replay

These commands recount source spans, dispositions and outcomes without model
weights or recordings. The two receipts include input commitments and numeric
measurements. A native replay requires the pinned source banks described in
the parent studies and the existing local model assets.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.audio_staggered_sine_transfer review \
  docs/evaluation-panels/audio-staggered-sine-2026-11-14.json
python3 -B -S -m scripts.audio_low_frequency_admission review \
  docs/evaluation-panels/audio-low-frequency-admission-2026-11-14.json
```
