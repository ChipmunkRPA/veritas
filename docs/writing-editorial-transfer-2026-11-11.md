# Short writing review on published journalism

The frozen short-passage selector found one more useful rewrite location than the current context check and flagged the same nine human files. The technical-prose reduction in false alerts did not carry over. This panel used eight articles from one publisher and supports keeping the local option at its existing scope. [Technical-prose comparison](writing-two-sided-usgs-transfer-2026-11-11.md)

## Sources and method

Eight separately bylined Global Voices articles from 2019 contributed excerpts on architecture, sport, music, the environment, film, history and theatre. The source pages credit the writers and display CC BY 3.0. The selected passages were originally published in English. Translation credits, captions, quotations and embedded media were excluded. Each controlled human replacement uses a genuine sentence from elsewhere in the same bylined article. The aggregate records source URLs, publisher positions and hashes without redistributing source or generated prose. [Publisher reuse policy](https://globalvoices.org/about/global-voices-attribution-policy/) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) · [Source and case records](evaluation-panels/writing-editorial-transfer-2026-11-11.json)

Each fixture contains four consecutive sentences and matches the local option's 68–134-word English plain-text scope. Three excerpts come from a single publisher paragraph. Five join adjacent publisher paragraphs with a space to fit the single-paragraph upload scope. There were no exact normalized 12-token overlaps between human source groups or with 45,348 retained project text entries. Model pretraining exposure remains unknown.

Two pinned Apache-2.0 generator families each attempted one sentence rewrite per group offline. All 16 attempts ended normally. Six passed the frozen length, sentence, copy, and scope rules, three from each generator. Five rejected attempts exceeded the target-copy allowance and five failed the word geometry rule. The original source roster, generator settings and admission rules were recorded and hashed before generation. Admitted cases and the same published fit were frozen before scoring. [Qwen terms](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM terms](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674) · [Pinned classifier](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

## Results

| Outcome | Current context | Current focus | Matched full window | Two-sided |
| --- | ---: | ---: | ---: | ---: |
| Useful AI rewrite locations | 2/6 | 1/6 | 3/6 | 4/6 |
| Useful locations, Qwen | 1/3 | 0/3 | 2/3 | 2/3 |
| Useful locations, SmolLM | 1/3 | 1/3 | 1/3 | 2/3 |
| Human originals flagged | 5/8 | 5/8 | 5/8 | 7/8 |
| Genuine replacements flagged | 4/8 | 4/8 | 4/8 | 6/8 |
| All human files flagged | 9/16 | 9/16 | 9/16 | 13/16 |
| Human source groups with a flag | 5/8 | 5/8 | 5/8 | 7/8 |
| Flags overlapping the genuine replacement | 3/8 | 3/8 | 4/8 | 3/8 |

Useful localization required at least 0.30 non-whitespace character intersection over union with the known edited sentence. The matched rule recovered three new locations and lost the two locations found by current context. The same nine individual human files triggered both methods. The two-sided variant found one more edit than the matched rule and flagged four additional human files.

The frozen runner made 18 CPU forwards over 142 unique complete contexts. A separate replay checked all 22 decoded fixtures, six edited locations, 40 exact publisher text locations, 264 context geometries, 120 outcome counts and four source-group counts. It also applied the serving selector to all 22 sets of saved responses and reproduced their matched selections.

The panel covers short, normalized excerpts and controlled sentence replacements. Whole articles, other publishers and naturally occurring editorial workflows need separate evidence. The serving option and overall confidence stay unchanged. [Frozen runner](../backend/scripts/writing_two_sided_transfer.py) · [Complete aggregate](evaluation-panels/writing-editorial-transfer-2026-11-11.json)
