# The detection loss persisted with shared river excerpts

The three local batches completed and replayed successfully under the
frozen resource guard, with 158 classifier forwards. The comparison kept
24 planned pairs. Nineteen pairs had both clips prepared and five were
unavailable. Each pair received the same early and late excerpts of one
river recording at 0 dB. Each clip had its own gain-matched clean control.
[Method](audio-demand-crossover-score-v214-method.md) · [Verified counts](evaluation-panels/audio-demand-crossover-score-complete-v214.json)

| Condition | Synthetic leaning AI | Synthetic leaning original | Synthetic inconclusive | Genuine leaning AI | Genuine leaning original | Genuine inconclusive | Genuine gate rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched clean | 16 | 2 | 1 | 0 | 19 | 0 | 0 |
| Early river, 0 dB | 1 | 3 | 15 | 0 | 11 | 7 | 1 |
| Late river, 0 dB | 0 | 2 | 17 | 1 | 8 | 9 | 1 |

Each row contains 19 prepared genuine and 19 prepared synthetic clips.
All prepared synthetic clips passed the speech gate. Of the 16 synthetic
clips flagged in matched clean audio, 13 became inconclusive and two
leaned original with the early excerpt. With the late excerpt, 15 became
inconclusive and one leaned original. One genuine clip went from leaning
original with the early excerpt to leaning AI with the late excerpt.

The shared-excerpt result confirms that the loss also occurs when the
same part of the river recording is assigned to both classes. The early
and late excerpts are from one tape, and the synthetic clips come from one
generator collection. The prepared foregrounds were already used in the
earlier diagnostic. The fresh panel adds a second generator, two different
recordings and new genuine sources.
