# Check whether short speech alerts survive changes to the audio input

The last loud-background holdout found a room recording that drew no
alert in isolation but triggered an alert on every genuine voice
mixed with it. This comparison applies two kinds of small changes
to the current two-second speech check's input and measures which
alerts survive. Prior research motivates testing whether processing
changes the evidence used by a detector.
[Earlier holdout](audio-short-edit-union-holdout-2026-11-15.md) ·
[Speech enhancement and anti-spoofing research](https://arxiv.org/abs/2603.14767) ·
[Frequency augmentation research](https://arxiv.org/abs/2408.06922)

## Comparison

The timing, thresholds and evaluation cases are frozen before the
new readings. The existing check selects the reported interval.
Each alternative decides only whether to retain that alert.

- **Frequency views.** Attenuate the lows and highs separately by up
  to 4 dB using smooth shelves at 650 Hz and 3,200 Hz. Each filter
  reads the entire six-second upload with reflected boundaries and
  returns samples at the same clock. Both filtered readings must
  exceed the existing two-second PCM point.
- **Shifted windows.** Read the original waveform 80 ms earlier and
  later where there are distinct source samples. At the beginning or
  end of the file, use the one distinct available shift. The usual
  speech admission and operating point apply to every distinct view.
- **Either check.** Keep the alert if it passes either of the two
  preceding rules.

The filters keep the voices mixed and change their frequency balance.
Every interval emitted by a rule is exactly the interval from the
current scan. A useful location covers at least half of the labeled
edit and 30% of the union of the detected and labeled intervals.
The receipt tracks removed genuine alerts and lost useful locations
separately. The test keeps the current operating point for all
views and uses no fit on the filtered scores.

## Sources

- The 32 noisy reader pairs and six isolated rooms from the previous
  result supply a diagnostic panel.
- The retained room models' source bank supplies 24 PhonemeDF groups
  and eight stock groups. Each has an untouched example, genuine
  splice, isolated room and two short synthetic edits from two
  generators per bank. The comparison uses the existing source-clock
  measurements for the unmodified PCM.
- The fresh panel selects additional SINE readers from the first
  192 MiB of shards 22, 23, 24, 26, 27 and 28 at the pinned publisher
  revision. It excludes the declared readers in all named earlier
  SINE trials. Matched original and edited clips use identical room
  background mixed 5 dB louder than the original foreground by RMS.
  The eight recorded AMI backgrounds are reused from an earlier study.
  They differ from the six diagnostic recordings.

The fresh inputs have new reader identities by publisher ID. Prior
project use of the retained recordings is disclosed by panel.
Cross-corpus speaker identity and pretraining overlap are unknown.
[Retained source roles and permissions](audio-short-edit-room-training-2026-11-15.md#source-roles-and-delivery-conditions)
· [SINE terms](https://huggingface.co/datasets/PeacefulData/SINE#license)
· [AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Results

| Panel | Useful edits, current and shifted | Genuine originals and splices flagged, current → shifted | Room-only flags, current → shifted |
| --- | ---: | ---: | ---: |
| Prior noisy SINE diagnostic | 1/32 | 10/32 → 8/32 | 0/6 → 0/6 |
| Retained PhonemeDF | 2/48 | 16/48 → 12/48 | 0/24 → 0/24 |
| Retained stock | 0/16 | 0/16 → 0/16 | 0/8 → 0/8 |
| Fresh noisy SINE | 1/32 | 1/32 → 1/32 | 0/8 → 0/8 |

Both shelves and the combined either-check rule produced the same
flags as the current scan across all 302 deliveries. The filter-bank
frontend already normalizes each band's mean and variance over the
window. This may explain some of the stability under gentle shelving.
The time views preserved all four useful locations and cleared six
genuine alerts in the retained panels. Both useful retained PhonemeDF
locations also overlapped a genuine control under the same timing
rule. The fresh panel's only ordinary alert survived. The fresh result
does not support general suppression of alerts.
[Measured comparisons](evaluation-panels/audio-short-edit-input-views-2026-11-15.json.gz)

## Local evidence

For eligible six-second PCM and 16-bit FLAC audio, a completed
short-window alert can now include the nearby readings. The browser
and PDF show whether the elevated response repeated, fell below the
same review point, or lacked sufficient speech in a nearby interval.
JSON retains the original alert, the exact source ticks, speech
coverage and PCM digest for each nearby input. Signed source replay
recomputes those input digests. The offline speech-coverage verifier
can also recompute the activity counts from the original file.

The additional read has a separate local time budget. Its absence
leaves the original alert available. These readings carry zero weight
in the overall confidence score and leave the selected clip intact.
Lossy delivery remains outside the new comparison's tested scope.

Through the local checkout flow, a genuine noisy WAV showed a varied
response around the original 2.000–4.000 s alert. A fresh edited FLAC
showed a repeated response around 4.000–6.000 s using the one available
earlier window. Its matched original WAV had no short-window alert.
The three signed reports verified against the original uploads, and
the coverage verifier reproduced the available speech sample counts.
All three analyses followed simulated USD 9.99 payments. No jobs started
before payment. [Local verification receipt](evaluation-panels/audio-short-nearby-app-2026-11-15.json)

## Reproduce

The receipt carries source hashes, the selected ranges, scalar
scores, and whether the shifted views had enough speech. The numeric
review uses standard Python and the checked-in parent receipts.

```sh
export PYTHONPATH=backend
module=scripts.audio_short_edit_input_views
receipt=docs/evaluation-panels/audio-short-edit-input-views-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

Use the [original bounded-prefix procedure](audio-sine-resynthesis-transfer-2026-11-14.md#replay)
with shards **22, 23, 24, 26, 27 and 28** and the same eight room PCM files.
Restore the exact `pcm/` inputs from the earlier
[room-panel capture](audio-short-edit-room-training-2026-11-15.md#reproduce-locally)
and the [preceding fresh union holdout](audio-short-edit-union-holdout-2026-11-15.md#reproduce).
The scoring command can measure the named panels together or in
disjoint batches.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$run/plan" "$cache" "$background"
"$runner" -B -m "$module" capture "$run/plan" "$voice_weights" "$diagnostic" "$room_inputs" \
  "$run/readings.json" fresh diagnostic phoneme stock
python3 -B -S -m "$module" combine "$run/plan" "$run/receipt.json.gz" "$run/readings.json"
```

To replay the local checkout fixture, supply the diagnostic and fresh
sealed directories containing the corresponding `pcm/` inputs:

```sh
"$runner" -B -m scripts.audio_short_nearby_app "$run/receipt.json.gz" \
  "$diagnostic" "$run/plan" "$run/app"
```
