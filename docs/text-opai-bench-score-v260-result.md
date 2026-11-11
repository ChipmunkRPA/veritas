# Passage localization on matched government-report rewrites

All four methods assessed all 60 inputs and covered every non-whitespace source position. The same 12 reports supplied 12 human seeds and 48 rewrites. All four methods returned identical highlight masks and coverage on every input. The canonical text and the current extracted text were identical on all 60 inputs. The local run completed 396 classifier and 105 encoder forwards. [Verified inputs](text-opai-bench-intake-v257-method.md) · [Frozen scoring method](text-opai-bench-score-v260-method.md) · [Exact aggregate](evaluation-panels/text-opai-bench-score-complete-v260.json)

| Publisher edit stage | Rewrites with any marked overlap | Rewrites meeting the useful localization criterion | Known regions recovered |
| --- | ---: | ---: | ---: |
| Qwen polish | 1 of 12 | 0 of 12 | 1 of 38 |
| Qwen compression | 1 of 12 | 0 of 12 | 1 of 70 |
| GPT polish | 1 of 12 | 0 of 12 | 1 of 38 |
| GPT compression | 1 of 12 | 0 of 12 | 1 of 70 |

Useful localization requires the marked and publisher-labeled regions to share at least 30% of their combined non-whitespace characters. A known region counts as recovered when the highlight covers at least half its non-whitespace characters. Across the rewrites, the methods highlighted 544 of 47,813 publisher-labeled characters. They falsely highlighted 140 characters on one of the 12 human seeds. A separate projection of the saved responses showed that all four rewrites with any true overlap came from that same report. The four hits therefore represent one source group.

This test shows weak transfer to the publisher's polished and compressed government-report passages at the existing research points. Combining the methods added no evidence on this panel. The publisher labels entire rewritten sentence regions, and these results do not cover emails or naturally co-written documents. [Benchmark provenance method](https://arxiv.org/html/2606.06481v1)
