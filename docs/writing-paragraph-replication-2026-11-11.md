# Isolated paragraph review on court opinions and prepared speeches

This local replication used four U.S. Supreme Court majority opinions and four prepared Federal Reserve Board speeches, all published in 2021 by eight distinct officials. The court excerpts came from the official PDF text layer. The speech excerpts came from the Board’s HTML. The intake excluded dissenting opinions, extended quotations, headers and references. Some court PDF text may retain typesetting artifacts. Two other opinions had no eligible excerpt and were excluded before scoring. None of the admitted original or human-edited excerpts shared an exact normalized 12-token fragment with 45,185 retained project text entries or another admitted work. The original classifier’s full pretraining and training exposure is unknown. [Court publication index](https://www.supremecourt.gov/opinions/slipopinion/20) · [U.S. government works](https://www.copyright.gov/title17/92chap1.html) · [Federal Reserve speech archive](https://www.federalreserve.gov/newsevents/2021-speeches.htm) · [Federal Reserve reuse terms](https://www.federalreserve.gov/disclaimer.htm)

Each source supplied a four-paragraph excerpt, a same-source whole-paragraph edit, and a same-source single-sentence edit. The surrounding source text stayed fixed. The edits were constructed by splicing exact publisher prose from elsewhere in the same work. Two Apache-2.0 generator families each made one attempt to rewrite the same interior paragraph. Eleven of sixteen responses qualified before scoring, including six Qwen and five SmolLM rewrites. Four exceeded the source-copy limit and one had an incompatible response format. These are synthetic mixed-authorship documents with known edit locations. [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

The study froze the current three-sentence review, the earlier stricter reference points, a one-paragraph selection based on current scores alone, and that same selection with an additional isolated-paragraph response. The last comparison called the shipped opt-in paragraph scorer with the exact local weights. The complete paragraph had to fit in one tokenizer window and be read as one input. The confirmation uses the same weights as the context response. No operating point was fitted to these sources. Useful localization required at least 0.30 non-whitespace character intersection over union with the known edit. [Prior article panel and local option](writing-article-context-transfer-2026-11-11.md)

## Results

The run completed 197 CPU forward calls on all 35 eligible files. Of 70 interior paragraphs, 35 qualified for the additional review. Every qualifying paragraph was scored in full with a verified tokenizer input. Six came from court cases and 29 from speeches. A separate bitset replay matched all 30 primary arm-by-rule counts and transitions, six source group counts, and the exact hashes of the added paragraph inputs. [Aggregate, source commitments and per-family dispositions](evaluation-panels/writing-paragraph-replication-2026-11-11.json)

| Outcome | Current context review | Single-paragraph selection | Selection with isolated review |
| --- | ---: | ---: | ---: |
| Court human files flagged | 3/12 | 1/12 | 0/12 |
| Court human source groups with a flag | 1/4 | 1/4 | 0/4 |
| Court rewrites usefully localized | 4/4 | 4/4 | 4/4 |
| Speech human files flagged | 12/12 | 11/12 | 8/12 |
| Speech human source groups with a flag | 4/4 | 4/4 | 4/4 |
| Speech rewrites usefully localized | 4/7 | 6/7 | 6/7 |
| All human files flagged | 15/24 | 12/24 | 8/24 |
| All rewrites usefully localized | 8/11 | 10/11 | 10/11 |

Narrowing the evidence to one paragraph recovered three useful locations on speeches and lost one previously useful location by selecting an unchanged paragraph. The separate isolated response cleared four further human-file flags and kept every useful location selected by the narrower rule. It introduced no new human flags. Both generator families were represented in both genres. The four eligible court rewrites included just one SmolLM rewrite. Of the seven speech rewrites, the isolated rule found all three Qwen rewrites and three of four SmolLM rewrites.

Every speech source group still produced a human flag. The gain on these courts therefore does not establish a general false-alarm improvement, and multiple views of a source are correlated. Keep the additional check as optional inspection evidence and keep the serving confidence calculation and ordinary selection unchanged. The frozen attribution record retains a descriptive local label for one speech. The aggregate records the publisher’s corrected title without changing the source bytes or scores. [Governor Bowman’s published speech](https://www.federalreserve.gov/newsevents/speech/bowman20210623a.htm)
