# Test shorter speech windows under louder background conversation

The current two-second review cannot tightly locate a very short edit.
We tested whether changing interval selection could recover those edits
while keeping the underlying speech model fixed. Recent work motivates
scoring coherent intervals and distinguishing content changes from
transitions. This trial uses the project's own code and frozen weights.
[Temporal decoding research](https://arxiv.org/abs/2609.10051) ·
[Boundary and interior scoring research](https://arxiv.org/abs/2609.25822) ·
[Genuine splice controls](https://arxiv.org/abs/2610.04479)

## Frozen comparison

Two existing short-edit models were trained on separate generators.
Their held room panels comprise 24 PhonemeDF source groups and eight
stock groups using two other generators. Each model reads 112
calibration deliveries and 288 held room deliveries. The fresh SINE set
has 16 additional declared readers selected before inference from four
previously unused 192 MiB publisher prefixes at shards 8, 10, 11 and 12.
Each reader contributes a matched original and published synthetic
cut-and-paste edit. Each pair is delivered dry and with competing room
speech mixed 5 dB louder than the foreground. Eight additional isolated
room controls give 72 fresh deliveries. These room excerpts were used
in earlier studies. Reader separation is based on publisher identifiers
and exact input hashes. Pretraining and cross-corpus identity overlap
remain unknown.
[Earlier room model sources and terms](audio-short-edit-room-training-2026-11-15.md#source-roles-and-delivery-conditions) ·
[SINE publisher and Apache-2.0 declaration](https://huggingface.co/datasets/PeacefulData/SINE#license) ·
[SINE editing method](https://arxiv.org/abs/2501.03805) ·
[AMI source terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

1. The fixed search checks 500, 750 and 1,000 ms intervals every
   100 ms using the original content and boundary scores.
2. The denser search halves the start spacing with the same widths.
3. Width calibration adds 250 and 350 ms intervals. It puts each
   width on its own ordinary-control scale before choosing a maximum.
4. Context subtracts the stronger mean content score from the two
   neighboring speech intervals, then applies the same width-calibration
   procedure. Each flank is 250–300 ms with a 50 ms gap.

Every interval requires at least half detected speech. Context also
requires that coverage in each flank. Each route and known acoustic
condition uses 16 separate ordinary calibration groups. The wider
point is the nearest-rank 90th percentile of the ordinary group maxima,
bounded below by the highest assessed isolated-background score. The
stricter point replaces the percentile with the ordinary maximum.
Candidate widths use separate medians and 90th-percentile spreads before
the final operating point accounts for choosing among widths. The
condition is supplied by the experimental plan. Deploying this method
would require a validated way to route real recordings by condition.

## Results

A useful location covers half the marked edit and at least 30% of the
union of the highlighted and marked intervals. A matched collision
means a genuine recording from the same group received a flag that
would independently satisfy that overlap on the edit clock.

| Fresh louder-background SINE, wider point | Useful /16 | Useful clear of the matched original | Ordinary flags /16 | Isolated-room flags /8 |
| --- | ---: | ---: | ---: | ---: |
| First fit, fixed or denser | 1 | 1 | 6 | 1 |
| First fit, width-calibrated | 0 | 0 | 7 | 1 |
| First fit, with context | 0 | 0 | 0 | 0 |
| Second fit, fixed or denser | 0 | 0 | 2 | 0 |
| Second fit, width-calibrated | 1 | 1 | 2 | 0 |
| Second fit, with context | 2 | 0 | 4 | 0 |
| Current two-second app scan, existing point | 1 | 1 | 0 | 0 |

All foreground cases were assessed. The app left four of the eight
room-only files unassessed. Its two flagged edited files included one
wrong time. Its grid can fit only five of the 16 labeled edits even
when given the correct clock and ignoring speech admission. Shorter
intervals can improve that geometric limit, but the measured score
response remains inconsistent. The two useful locations from the
second fit's context check each also appeared on a matched ordinary
recording under the overlap rule.

Denser starts alone added no useful location on any louder-background
panel. At the stricter point in the earlier PhonemeDF recordings, the
first fit's width calibration added two useful locations, moving from
1/24 to 3/24. Ordinary flags moved from 35/72 to 34/72, while flags on
isolated backgrounds rose from 6/24 to 8/24. Width calibration also
added two correctly timed first-fit stock locations at the same 7/16
ordinary flags. Both stock gains coincided with a matched genuine
alert. The second fit lost one useful stock location at that point.
The fresh panel shows gains on different readers across fits and no
consistent improvement over the current app's false alarms. All
experimental routes remain outside serving and confidence.
[Frozen readings](evaluation-panels/audio-short-edit-duration-transfer-2026-11-15.json.gz) ·
[Current scan receipt](evaluation-panels/audio-short-edit-duration-serving-2026-11-15.json)

An exploratory union of the current scan and the second fit's wider
width-calibrated result located two different edits out of 16 and
flagged two ordinary originals out of 16. The union was considered
after seeing the results. A separately frozen comparison would be
needed to assess that combination. That [subsequent frozen comparison](audio-short-edit-union-holdout-2026-11-15.md)
tested 32 other reader pairs and observed zero additional useful
locations and four additional ordinary flags.

## Reproduce

The receipts contain public source identifiers, hashes, sample intervals
and scalar maxima. They exclude media, model weights and frame features.
The portable commands require only standard Python. A separate recount
matched 96 held result rows and 32 numerical operating points. Repeat
native runs reproduced both receipts byte for byte. A second source
intake reproduced the frozen plan and fresh PCM byte for byte.

```sh
export PYTHONPATH=backend
module=scripts.audio_short_edit_duration_transfer
receipt=docs/evaluation-panels/audio-short-edit-duration-transfer-2026-11-15.json.gz
current=docs/evaluation-panels/audio-short-edit-duration-serving-2026-11-15.json
python3 -B -S -m "$module" review "$receipt"
python3 -B -S -m scripts.audio_short_edit_duration_serving review "$receipt" "$current"
```

For native replay, follow the [original bounded prefix download](audio-sine-resynthesis-transfer-2026-11-14.md#replay)
with shards **8, 10, 11 and 12**. Use its eight verified background
excerpts. Restore the [original room plan and fitted assets](audio-short-edit-room-training-2026-11-15.md#reproduce-locally)
in a directory containing the sealed input and adjacent `fit/` folder.
The main freeze copies only the held and calibration inputs needed
by this experiment.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$run/plan" "$cache" "$background" "$sealed_room"
"$runner" -B -m "$module" calibration "$run/plan" "$voice_weights/silero_vad.jit"
"$runner" -B -m "$module" held "$run/plan" "$voice_weights/silero_vad.jit" "$run/duration.json.gz"
"$runner" -B -m scripts.audio_short_edit_duration_serving capture "$run/plan" "$voice_weights" "$run/current.json"
```
