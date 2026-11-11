# The rollback classifier ranked noisy speech less accurately

The local comparison completed all 112 eligible windows across three
conditions. The current and rollback classifiers used the same admitted
speech windows. Rank concordance measures how often a synthetic clip
scores higher than a genuine clip, with ties counted as half.
[Method](audio-rollback-crossover-v220-method.md) · [Verified counts](evaluation-panels/audio-rollback-crossover-complete-v220.json)

| Condition | Current raw rank concordance | Rollback raw rank concordance |
| --- | ---: | ---: |
| Matched clean | 0.992 | 0.950 |
| Early river, 0 dB | 0.792 | 0.506 |
| Late river, 0 dB | 0.722 | 0.632 |

On the 18 genuine–synthetic pairs that shared each noisy excerpt, the
current classifier ranked synthetic higher on 13 early pairs and 11 late
pairs. The rollback ranked synthetic higher on eight early pairs and 11
late pairs. The rollback also lowered all 19 synthetic scores relative to
matched clean audio in each noisy condition.

Retain the current classifier for this route. The rollback had weaker
separation in both noisy conditions and no useful recovery at its
historical diagnostic point. This comparison covers previously assessed
Turkish excerpts and one river recording. The fresh two-generator
comparison used separate kitchen and traffic recordings.
[Fresh panel results](audio-fresh-shared-noise-score-v219-result.md)
