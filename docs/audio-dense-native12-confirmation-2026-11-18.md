# Independent confirmation of the denser earlier-copy audio check

The fixed candidate added 30 correctly located noisy replacements on new
readers, with zero added ordinary-control highlights or misplaced synthetic
ranges. Every prospective advancement condition passed. The local application
now uses the candidate in an optional earlier-copy comparison.

## Source independence and fixed method

This confirmation uses 16 original LibriSpeech recordings from eight newly
qualified declared readers. A conservative audit joined 632 retained source
metadata files and excluded 818 consumed reader IDs, known source roles,
transcripts and eight-word openings. The retained archives had insufficient
unused readers. One additional 463.8 MB publisher-pinned Parquet supplied the
new sources. Its complete SHA-256 was verified twice on the external drive.
Nineteen predeclared native checks admitted 16 hosts. An alternate decoder
reproduced every original lossless source and twelve-second fixture.
[LibriSpeech publisher and CC BY 4.0 terms](https://www.openslr.org/12/)
· [Pinned official dataset object](https://huggingface.co/datasets/openslr/librispeech_asr/resolve/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/clean/train.360/0004.parquet)

Three recorded DEMAND sessions supply appliance, river and restaurant
backgrounds. These physical captures and families differ from the development
panel's traffic and AMI recordings. They have earlier unrelated project
exposure. Their exact twelve-second selections were frozen before measurement.
The scope is a prospective source confirmation for this candidate. Unknown
speaker aliases, pretrained-data exposure and unrestricted recording conditions
remain open limitations.
[DEMAND publisher and CC BY-SA 3.0 terms](https://zenodo.org/records/1227121)

The candidate keeps the current twelve-second result first. When the existing
reference and frequency-balance requirements permit a fallback, it applies
the existing dense 960 ms measurement at six fixed six-second starts and
requires agreement from neighboring windows. Its decision point, source
clock, matching requirements and frequency guards stayed fixed. The comparison
receives the supplied clean earlier copy and submitted recording. It loads
zero detection models and receives zero AI-confidence weight.
[Method and unchanged 624-pair development comparison](audio-dense-native12-reference-2026-11-18.md)

Each host supplied two 960 ms synthetic inserts from the existing two stock
families, two genuine splices, two local resampling controls and an unchanged
copy. New authored prompts were frozen before 32 generator calls. Both voices
and generator families had development exposure. The noise was 10 dB above the
original target speech. The same 32 target events recur across conditions.
Two hosts per reader and paired genuine donors form four connected components.

## Confirmed results

A correct location covers half the known replacement and reaches 0.30 temporal
intersection over union. Genuine splices are physical-change positives. The
96 ordinary negatives per condition comprise unchanged copies, local
roundtrips and whole-recording gain or frequency changes.

| Condition | Synthetic locations, current / candidate | Genuine splice locations, current / candidate | Ordinary highlights, current / candidate | Synthetic files with a misplaced highlight, current / candidate |
| --- | ---: | ---: | ---: | ---: |
| Clean | 13 / 26 of 32 | 14 / 26 of 32 | 0 / 0 of 96 | 0 / 0 of 32 |
| Appliance, -10 dB | 14 / 26 of 32 | 14 / 26 of 32 | 0 / 0 of 96 | 0 / 0 of 32 |
| River, -10 dB | 8 / 20 of 32 | 8 / 20 of 32 | 5 / 5 of 96 | 0 / 0 of 32 |
| Restaurant, -10 dB | 4 / 10 of 32 | 4 / 10 of 32 | 14 / 14 of 96 | 4 / 4 of 32 |

The noisy totals cover 96 deliveries from 32 paired target events. Correct
locations rose from 26 to 56. The 30 additions span all four connected
components and both generator families. All previous ranges were retained.
The 19 inherited ordinary highlights and four inherited misplaced restaurant
highlights remain visible in both arms.

Reference matching left zero clean or appliance, eight river and twelve
restaurant synthetic comparisons unassessed in each arm. The candidate leaves
six clean, six appliance, twelve river and 22 restaurant targets without a
correct location. These counts include unassessed comparisons.

The 1,264 pairs also include 192 unrelated, shifted or half-matched earlier
copies, 128 intact recordings with a half-percent clock change, 48 isolated
backgrounds and 256 synthetic comparisons with the earlier copy at half or
1.4 times its original level. Wrong-reference, clock and isolated-background
controls produced zero highlights. At both earlier-reference levels, the
correct-location counts match the principal table in every condition.
These are repeated robustness comparisons of the same target events.

## Prospective error budget and interpretation

The sealed plan required four added noisy locations across two components and
both families. It allowed up to four added ordinary rows and one newly flagged
ordinary component per condition, and one added misplaced synthetic range per
condition. Wrong-reference, clock and isolated-background flags had to remain
zero. The candidate met every requirement with zero added ordinary or misplaced
highlights. This budget was fixed before generation or measurement.

The earlier development comparison keeps its original results and stricter
failed gate. Its ten added noisy locations, four new ordinary traffic-control
highlights from one host and one new misplaced clean synthetic highlight stay
part of the evidence. The confirmation measures the same candidate on separate
host readers and background sessions under the prospectively stated tradeoff.

This check establishes changes relative to a supplied earlier recording.
Genuine editing and ordinary processing can create the same kind of evidence.
AI authorship and individual-speaker attribution require separate support.
The earlier standalone voice-check results remain a separate experiment.

## Native replay and public reproducibility

Before outcome measurement, a separate implementation reconstructed all 1,264
source pairs, 2,528 source clocks, 32 generated phrases and selected fragments,
and the exact source-relative noise intervals. The target SNRs ranged from
-10.0002 to -9.9999 dB. Outcome replay independently recomputed both physical
measurements on all pairs and reproduced 2,838 dense-window records, each
selected range, admission state, clock and grouped count. It used its own
measurement code and the same pinned numerical filtering libraries.

Required generator-family membership is unordered. A list-order-only tally
error was identified and clarified before closure. The original result and
measurement journal were preserved. Independent recount applies the original
set-membership requirement. Inputs, measured decisions, points and error
budgets stay unchanged. Ten authored scalar tests checked the count and gate
logic independently of held outcomes.

The [public aggregate](evaluation-panels/audio-dense-native12-confirmation-2026-11-18.json)
provides grouped condition and role counts, the prospective error budget,
publisher source citations and native custody commitments. Media, individual
decisions, scores, reader IDs, transcripts, local paths and individual edit
positions remain private. The standard-library reviewer recounts the tables
and advancement conditions. Connected-component counts are native-audited
aggregate inputs. Reproducing individual source evidence uses the retained
private pairs and frozen recipes.

```sh
python3 -B -S backend/scripts/review_audio_native12_dense_confirmation.py \
  docs/evaluation-panels/audio-dense-native12-confirmation-2026-11-18.json
```

## Local application

On a development instance, enable `VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH`,
`VERITAS_AUDIO_REFERENCE_LONG_CLEAN_SPEECH` and
`VERITAS_AUDIO_REFERENCE_DENSE_LONG_CLEAN_SPEECH` with audio reference
comparison available. From a completed audio analysis, supply an earlier
copy and confirm it is clean. This check requires two continuous, exactly
twelve-second, 16 kHz mono PCM16 WAV or FLAC recordings on the same clock.

The page and PDF show the selected interval, both waveforms and the starts
of the two supporting windows. The JSON also includes exact sample ranges
and both playable excerpts. `scripts.compare_audio_references` and
`scripts.verify_audio_clean_speech_export` verify the record against the
two original files. Historical exports still replay with their original
method. The comparison carries zero weight in the standalone AI assessment.

The serving method reproduced all 1,264 frozen comparison outcomes, including
192 retained ranges and 171 ranges from its fallback. Nine simulated prepaid
analyses exercised added locations in all three noise conditions, unchanged
controls, an incorrect reference, a clock mismatch and a previously flagged
ordinary recording. Chromium verified all 18 PDF and JSON downloads and the
bytes and duration of all eight playable excerpts. The comparison created zero
additional analysis jobs and preserved each standalone assessment.
