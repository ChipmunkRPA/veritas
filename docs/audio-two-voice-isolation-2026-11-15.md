# Short speech edits with exact isolated voices

The earlier two-voice benchmark measured what the detector did on the
original mix and on estimated stereo voices. This follow-up uses each
voice's exact isolated track to test how much of the failure comes from
speaker overlap, room filtering, or the short-edit classifier. It also
looks directly at the known edit time to separate finding the interval
from scoring the speech inside it. Partial-spoof research makes a similar
distinction between boundary detection and segment classification.
[Two-voice benchmark](audio-two-voice-components-2026-11-15.md) ·
[Boundary and segment research](https://arxiv.org/abs/2604.02913)

## Inputs

Reuse all 28 previously frozen scenes and their exact stems. Each scene
contains two voices and each voice has an untouched, genuinely edited,
and synthetically edited version. This gives 72 distinct development
voice versions and 96 held versions. Among the held versions are 32
synthetically edited voices, 32 genuinely edited voices, and 32 untouched
voices. The held PhonemeDF and stock-voice cohorts each supply 16 of every
type. The two held cohorts use the same BBC room response with different
speakers and four generator families between them. Parent receipts retain
the source reuse, ancestry, room placement, and commercial-use terms.

The classifier and speech activity gate see one second of speech at a time.
The four views are the original mono mix, spatial estimates matched to the
known voice after inference, the exact isolated room-rendered voice, and
the exact dry voice on its planted sample clock before the room response.
The dry and room views come from the same speech, with different acoustic
processing and delivered levels. The scorer uses the same pinned
classifier across all four. The current two-second app scan separately
runs on the exact rendered voices at its existing point.
[Pinned classifier and terms](https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification/tree/f7050b586236dc910d1157f430def2d0647b02b4) ·
[Current model terms](commercial-model-audit.md)

## Matching and calibration

For a directional estimate to stand in for one voice at a given time,
its scale-invariant signal-to-distortion ratio must be at least 1 dB
closer to that voice than to the other. When several directional
estimates qualify at the same time, select the closest measurement
with an admitted classifier score. This matching uses the benchmark's
known stems and is available as a diagnostic. The assigned estimate may
change from window to window.

Each one-second view has a separately fitted point. The strict point is
its highest genuine response, and the wider point is the second-highest
speaker-pair maximum. Only the untouched and genuinely edited development
scenes set these points. Held measurements use strict excess above the
points fixed in development. Source-level false flags are counted on
both the genuinely edited and untouched versions. The mono and estimated
views are evaluated for each voice in its single-edited condition and
against the corresponding ordinary controls. A mono reading can be
counted once for each voice on the same ordinary delivery. These are
source-level counts, distinct from unique-file false alarms.

A separate diagnostic supplies the detector bookkeeping with the known
edit location. Select the one-second grid window with the highest interval
overlap, taking the earliest tie before seeing scores. At that fixed time,
compare the synthetic insertion with its genuine control twice, with the
competing voice genuinely edited and with it synthetically edited. The
isolated views are identical between those two conditions. This comparison
does not select or calibrate a customer report interval.

## What the isolated tracks recover

Every synthetic voice had an admitted window in all five checks. The
strict one-second points yielded zero useful ranges across all four
views. The table shows the wider one-second point alongside the app
scan's existing point. The denominator for useful ranges is 16 edited
voices per cohort. The ordinary column counts 32 voice observations per
cohort, equally divided between untouched and genuinely edited tracks.

| Input and scan | PhonemeDF useful ranges | PhonemeDF ordinary flags | Stock useful ranges | Stock ordinary flags |
| --- | ---: | ---: | ---: | ---: |
| Mono mixture, one second | 0/16 | 12/32 | 1/16 | 12/32 |
| Estimate matched to the known voice, one second | 0/16 | 0/32 | 0/16 | 0/32 |
| Exact room-rendered voice, one second | 0/16 | 6/32 | 0/16 | 6/32 |
| Exact dry voice, one second | 2/16 | 0/32 | 0/16 | 5/32 |
| Exact room-rendered voice, current two-second scan | 0/16 | 0/32 | 3/16 | 1/32 |

The app scan's three isolated-track locations were stock edits from
the same generator. They were all new relative to the same scan on
the corresponding single-edited mixes. The isolated scan also lost
the mix's one useful location in that comparison. The app's two-second
grid was too broad for eight of the 16 PhonemeDF voice edits. The
experimental one-second grid could reach all 32 held edits.

Results at the known edit clock show a separate failure. Each cell
counts edits scoring higher than their matched genuine edit in both
competing-voice conditions, over the sources with all four readings.

| Input | PhonemeDF | Stock |
| --- | ---: | ---: |
| Mono mixture | 4/16 | 7/15 |
| Estimate matched to the known voice | 8/16 | 9/14 |
| Exact room-rendered voice | 9/16 | 9/14 |
| Exact dry voice | 14/16 | 9/14 |

The drop from 14 to nine on PhonemeDF remains when the competing voice
is removed. This comparison changes both the room response and the
rendered level. The stock bank does not repeat the drop. The stricter
points also show a localization problem even when isolation is exact.
These results support giving short-edit calibration paired dry and
room-rendered speech, with genuine edits from the same source as controls.
The labels and fixed-clock diagnostics use reference stems available in
this benchmark. The evaluation adds no speaker attribution to customer
analyses.

## Replay

The portable reader verifies the exact parent receipt and recomputes the
development cutoffs, useful ranges, false flags, coverage, and edit-clock
comparisons. The package contains numeric readings, sample positions and
hashes. Source recordings, reconstructed speech, embeddings and model
weights stay in the local source setup.
An independent recount reconstructed 72 location and fixed-clock outcome
rows from the scalar responses, including all frozen cutoffs.

```sh
export PYTHONPATH=backend
module=scripts.audio_two_voice_isolation
receipt=docs/evaluation-panels/audio-two-voice-isolation-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

For native reproduction, use the preceding benchmark's `freeze` command
to rebuild the exact local inputs and their source stems. Retain the
PhonemeDF dry-source folder and the stock-voice native archive from that
step. Supply the pinned one-second classifier, speech activity file and
app voice asset directory from the earlier guide. The runner verifies the
original dry PCM and rendered stem hashes before scoring them.
[Parent source reconstruction](audio-two-voice-components-2026-11-15.md#replay)

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" measure development "$inputs" "$run/dev" "$phoneme" "$stock" "$classifier" "$speech"
"$runner" -B -m "$module" measure held "$inputs" "$run/held" "$phoneme" "$stock" "$classifier" "$speech" \
  --development "$run/dev/scores.json"
"$runner" -B -m "$module" current "$inputs" "$run/app.json" "$voice_weights"
python3 -B -S -m "$module" combine "$run/dev/scores.json" "$run/held/scores.json" \
  "$run/app.json" "$run/receipt.json"
```
