# Paragraph evidence on a new article panel

We checked 25 eLife Insight articles from November and December 2021 and November 2022. Eleven works qualified under an explicit article-level CC BY 4.0 notice and the fixed excerpt rules. Fourteen works had no suitable excerpt. The admitted works shared no authors or linked research DOIs with the checked earlier eLife records. The normalized 12-token overlap check found no matches across 44,984 text entries in 30 retained packets. Model pretraining exposure is unknown. The human arms comprise 11 unchanged excerpts, one deterministic copyedit, and nine same-article paragraph substitutions. [Publisher terms](https://elifesciences.org/terms) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The two Apache-2.0 generator families each attempted one interior paragraph rewrite per article. The plan raised the generation budget to 384 tokens, asked for a more explicit target length, and required a genuine end-of-sequence. It retained the existing single-paragraph, length, source-copy, wrapper and source-binding checks. Every admitted replacement leaves the surrounding text unchanged. The exact aggregate includes publisher attribution and disposition counts. The comparison to the earlier 200-token panel uses different articles, prompts and decoding settings. [Earlier local transfer](text-editorial-transfer-2026-11-11.md) · [Exact aggregate](evaluation-panels/writing-paragraph-context-transfer-2026-11-11.json)

## Comparison

The baseline reuses the local classifier's native three-sentence context, exact input coverage, and previously fixed raw and stricter review points. The candidate scores each paragraph independently using the same pinned model and compares each eligible interior paragraph with its two immediate neighbors. Each of the three paragraphs must have at least 20 words. One rule uses the isolated response. A second rule uses its log-odds margin above the stronger neighbor. A third requires both points on the same paragraph. A candidate can highlight one unique qualifying paragraph. The highlight covers the exact scored paragraph. The two points were fixed using the second-largest file maximum from 13 earlier development originals. No result on the new sources entered the point selection.

The design takes the use of nearby evidence from the LA-CPD paper and pairs originals with their edited variants following the comparison logic in ARB. The local implementation uses paragraph-level response contrast and matched copyedit and splice controls. It uses none of the papers' code, weights, or data. A useful location overlaps the known replacement with at least 0.30 non-whitespace character intersection over union. [LA-CPD](https://arxiv.org/html/2609.33787v1) · [Matched Authorship-Rewriting Benchmark](https://arxiv.org/html/2607.29539v1)

## Eligibility and local results

The primary 22 attempts all ended within the 384-token limit. Five Qwen rewrites qualified. All 11 SmolLM outputs were excluded under the original format, word-count, and source-copy checks. A separately frozen preparation rule inspected the raw output blocks before any confirmation score. It kept three additional Qwen paragraphs by taking the longest complete sentence prefix within the original word bounds. Every considered SmolLM block copied too much from the source or was unchanged. The primary preparation dispositions remain in the aggregate.

A separately frozen target-only prompt for SmolLM asked for shorter prose in the model's own wording. Ten of 11 attempts passed the unchanged admission checks. The excluded response copied too much from the source. The combined comparison therefore covers 18 eligible rewrites across 11 new source articles, with eight Qwen and ten SmolLM files. Seven articles supplied eligible files from both families. Both prompts and the response preparation are documented separately because they affect the types of rewrites the detector sees.

| Outcome | Raw review | Strict raw | Isolated paragraph | Two-sided contrast | Both paragraph checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Human originals flagged | 10/11 | 2/11 | 3/11 | 1/11 | 1/11 |
| Human copyedits flagged | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| Same-article human substitutions flagged | 8/9 | 2/9 | 4/9 | 2/9 | 2/9 |
| All human controls flagged | 18/21 | 4/21 | 7/21 | 3/21 | 3/21 |
| Qwen rewrites usefully localized | 5/8 | 2/8 | 4/8 | 3/8 | 3/8 |
| SmolLM rewrites usefully localized | 6/10 | 2/10 | 5/10 | 3/10 | 3/10 |
| All eligible rewrites usefully localized | 11/18 | 4/18 | 9/18 | 6/18 | 6/18 |

Compared with the strict raw rule, requiring both paragraph checks added three useful locations, retained three, and lost one. It cleared three flags on human files and introduced two. Both families gained one net useful location. The isolated response alone recovered five more locations than strict raw and added three human flags. The broader raw review still found the most edits and flagged most human controls. All 39 evaluated files had complete scored coverage for every rule. The development run made 26 local classifier forwards. The three completed confirmation receipts account for 331 forwards.

The combination is a candidate for another independent panel. This comparison has eleven source articles, one copyedit, and deliberately constructed single-paragraph rewrites. The two generator families used different final preparation methods. The two paragraph measurements share the same detector weights. The customer verdict and confidence are outside this experiment.

The first standalone score of the three recovered cases wrote all responses, then stopped because its output directory lacked the point receipt. A corrected rerun used the exact sealed point and reproduced the responses byte for byte. The first target-only launcher stopped before loading a model because it imported a CPU-only dependency inside the isolated generation runtime. The corrected launch preserved its frozen prompt and eligibility. Raw inputs and case-level outputs stay local. [Source attribution, exact counts and execution receipts](evaluation-panels/writing-paragraph-context-transfer-2026-11-11.json)

The [portable arithmetic](../backend/app/bench/paragraph_context.py) takes paragraph spans, word counts and scalar responses. It reselected the exact saved spans in all 117 decisions across the three paragraph rules and 39 evaluated files. The authored checks can run with no model or media using `PYTHONPATH=backend backend/.venv/bin/pytest -q backend/tests/test_paragraph_context_bench.py`. The authored examples verify the calculation and selection behavior. The full result also depends on the private source qualification and exact model responses.
