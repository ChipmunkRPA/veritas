# Passage contrast improved ranking on reports without useful localization

The model-free replay compared three passage masks across 388 planned inputs from 44 source groups. Of those inputs, 326 had a saved score and 62 earlier article controls remained unavailable from preparation. The fixed contrast cutoff came from the 128 all-human v7 compositions, where the current and contrast points each highlighted one of 32 groups. None of the six pairwise comparisons between the point and evaluation cohorts found a shared normalized 12-token fragment. [Frozen method](text-context-shift-v266-method.md) · [Exact aggregate](evaluation-panels/text-context-shift-complete-v266.json)

| Evaluation | Current | Context shift | Union |
| --- | ---: | ---: | ---: |
| Report rewrites with any labeled overlap | 4 / 48 | 4 / 48 | 8 / 48 |
| Report rewrites with useful localization | 0 / 48 | 0 / 48 | 0 / 48 |
| Original reports falsely highlighted | 1 / 12 | 1 / 12 | 2 / 12 |
| Article rewrites and inserts with useful localization | 3 / 131 | 0 / 131 | 3 / 131 |
| Short article inserts with useful localization | 0 / 23 | 0 / 23 | 0 / 23 |
| All-human article originals and formatting controls flagged | 2 / 131 | 8 / 131 | 10 / 131 |

The report contrast improved within-input ranking on the 12 source groups from 0.533 to 0.591, giving ties half credit. On the separate article cohort, the same comparison on 30 groups fell from 0.727 to 0.648. The shift measured every source position in the reports. It fully assessed 248 article inputs, partially assessed 11, and left three scored inputs unassessed. The four human revision excerpts offered only 32 of 621 character positions to the shift, all in one excerpt. Its zero flags on those excerpts carry that limited coverage. [Report baseline](text-opai-bench-score-v260-result.md) · [Article baseline](text-gohumanize-transfer-v196-result.md)

The shifted score is sensitive to both the sentence entering the context and the one leaving it. The extra report overlaps failed the fixed useful-location criterion, and the union falsely highlighted eight additional all-human article inputs. Keep the shift out of the serving path.
