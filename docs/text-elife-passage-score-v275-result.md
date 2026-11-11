# Passage localization on science-explainer rewrites

Neither method usefully localized any of the **six eligible rewrites** at the frozen research points. The current check covered all five originals and all six mixed articles but highlighted no passages. The interval method touched two known rewrites, one from each generator. Its highlights also covered **3,494 unchanged characters**, so neither reached the required character overlap. Neither method highlighted the five human controls. The interval method gave only partial coverage on four of those controls and all six mixed articles. Four additional planned rewrites were unavailable from preparation. [Frozen method](text-elife-passage-score-v275-method.md) · [Exact results](evaluation-panels/text-elife-passage-score-complete-v275.json)

| Outcome | Current | Interval |
| --- | ---: | ---: |
| Fully assessed, of 11 prepared articles | 11 | 1 |
| Mixed articles with any true overlap | 0/6 | 2/6 |
| Useful locations by character IoU | 0/6 | 0/6 |
| Human originals with highlights | 0/5 | 0/5 |
| Unchanged characters highlighted in mixed articles | 0 | 3,494 |

On positions covered by both methods, the current raw response ranked generated characters above human characters with mean concordance **0.841 across five rewrites from one family** and **0.920 for the single other rewrite**. The interval method's corresponding values were both 0.500. A follow-up checked whether the target paragraph actually ranked first, including on matched originals. This small cohort supplies no new calibration or basis for replacing the current check. The two local batches completed 209 forwards, and a separate readout reproduced the aggregate exactly. [Paragraph rank diagnostic](text-elife-paragraph-rank-v281-result.md) · [Source and construction limits](text-elife-rewrites-v273-result.md)
