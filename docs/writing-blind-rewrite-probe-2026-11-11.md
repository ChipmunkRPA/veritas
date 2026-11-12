# Testing local copyediting to locate a partial rewrite

The earlier-draft comparison can highlight exact text changes when both drafts are available. This experiment checks one submitted version. It asks whether two small language models make fewer changes to a rewritten sentence than to its neighboring sentence when asked to copyedit each separately. RAIDAR and Learning to Rewrite report this pattern at longer document scales. This comparison uses new local code, commercially usable checkpoints and original test fixtures. [RAIDAR paper](https://arxiv.org/abs/2401.12970) · [Learning to Rewrite paper](https://arxiv.org/abs/2408.04237) · [Matched authorship-rewriting evaluation](https://arxiv.org/abs/2607.29539)

## Sources and method

Eight 2019–2020 National Park Service news releases supplied four-sentence excerpts and ordinary human prose controls. The releases represent six park offices and two topics from the national communications office. Each group supplied an unmodified excerpt and a version with a length-matched sentence from elsewhere in the same release substituted at the target. These substitutions are constructed controls for an ordinary change in context. The excerpts used NPS-authored prose without third-party quotations, image captions or navigation. Exact normalized 12-word comparisons found zero overlap with 45,457 retained entries from 61 prior project source packets. [NPS materials policy](https://home.nps.gov/aboutus/disclaimer.htm) · [Publisher sources and anchors](evaluation-panels/writing-blind-rewrite-probe-2026-11-11.json)

Two pinned Apache-2.0 instruction checkpoints attempted one target rewrite per source, with one published interior sentence selected per source before generation. Eleven of sixteen generated sentences met the fixed lexical, length and source geometry requirements. The output was verified as exactly one replacement sentence, with the three other sentences unchanged. Some accepted rewrites simplified or altered source details. The admission requirements assess the replacement geometry and lexical change. [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM2 license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/LICENSE)

Each probe saw one of the two eligible interior sentences at a time. A probe picked a sentence only if its normalized word edit distance was at least 0.15 smaller than the other sentence's distance. The agreement arm required both probes to pick the same sentence. The baseline used the existing raw three-sentence classifier evidence, capped at six windows. A useful location overlaps the exact non-whitespace source characters of the known edit with intersection over union of at least 0.30. Human false flags are counted separately for each fixture and once per source group. Models were loaded one at a time. No customer data or external model endpoint was involved. [Local runner](../backend/scripts/writing_blind_rewrite_probe.py) · [Current classifier license](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

## Local results

| Outcome | Current context | Qwen probe | SmolLM2 probe | Both probes agree | Current or both |
| --- | ---: | ---: | ---: | ---: | ---: |
| Useful location in Qwen edits | 0/5 | 4/5 | 0/5 | 0/5 | 0/5 |
| Useful location in SmolLM2 edits | 0/6 | 3/6 | 1/6 | 0/6 | 0/6 |
| Untouched human excerpts flagged | 0/8 | 5/8 | 3/8 | 1/8 | 1/8 |
| Human prose substitution excerpts flagged | 1/8 | 3/8 | 2/8 | 0/8 | 1/8 |
| Distinct human groups flagged | 1/8 | 6/8 | 4/8 | 1/8 | 2/8 |

Both models produced valid copyedits for both foci in five of eleven AI-edited cases and eleven of sixteen human controls. Each probe also served as an edit generator, so the table separates same-family and cross-family cases. The Qwen probe located three edits from the other family, but its eight flags on human controls include five untouched files. The agreement rule recovered no known edit. These results do not support deploying the probe or adding it to confidence. They cover one publisher and controlled single-sentence changes in English.

The current checkpoint completed 23 CPU forwards over 89 unique context inputs. Each probe read 35 unique source sentences. The independent verification rebuilt 40 publisher sentence ranges, checked all 27 complete fixtures, 54 exact focus ranges and 108 current context ranges, and recomputed 108 probe distances and 64 outcome counts. [Aggregate and commitments](evaluation-panels/writing-blind-rewrite-probe-2026-11-11.json) · [Independent verification script](../backend/scripts/review_writing_blind_rewrite_probe.py)
