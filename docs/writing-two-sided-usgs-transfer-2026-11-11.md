# Fixed sentence checks on USGS technical prose

The matched full-window fit found five of nine rewritten passages and flagged three of fourteen human files. The current check found four rewrites and flagged six human files. The two-sided fit found four rewrites and flagged six human files. This separate publisher transfer used the exact earlier coefficients and operating points. [Prior NIST comparison](writing-two-sided-context-2026-11-11.md) · [Full USGS aggregate](evaluation-panels/writing-two-sided-usgs-transfer-2026-11-11.json)

## Sources and method

Seven distinct U.S. Geological Survey publication-author groups supplied dated technical prose from 2001–2020. Topics included land imaging, glacier retreat, well monitoring, hydrographic surveys, landslides, land subsidence and river monitoring. Three publications have a named individual author and four have distinct coauthor groups. Each example uses four consecutive sentences from one body paragraph. Each ordinary replacement is an exact sentence from a different body paragraph of the same publication. The intake used attributable USGS prose and excluded images, captions, acknowledgments and quoted third-party material. The aggregate records the source bytes, paragraph and codepoint commitments and the attribution. [USGS rights policy](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)

The originals and ordinary replacements had zero exact normalized 12-token matches against 45,325 historical text entries or between the new groups. Six source files were PDFs with indexed text blocks and one was an HTML page with an indexed paragraph. Normalization removed soft hyphens and collapsed whitespace before freezing the exact original and donor spans. Model pretraining exposure is unknown.

The two offline, pinned Apache-2.0 generators attempted one rewrite per source group. Four of seven Qwen outputs and five of seven SmolLM2 outputs met the unchanged single-sentence, length, copy and overlap checks. Three answers failed length, one failed sentence structure and one copied too much of its target. All retained replacements preserved every other source character and had an eligible interior focus with exact neighbors. [Qwen terms](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM2 terms](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

The classifier measured the same native input geometries as the NIST comparison. The two fitted arms loaded the exact published fit artifact. Neither the authors nor the rewrites entered fitting or point selection. A useful range requires at least 0.30 intersection over union with the known rewrite, measured in non-whitespace characters.

## Results

| Outcome | Current context | Current focus | Matched full-window fit | Two-sided fit |
| --- | ---: | ---: | ---: | ---: |
| Human originals flagged | 4/7 | 4/7 | 2/7 | 3/7 |
| Genuine sentence replacements flagged | 2/7 | 2/7 | 1/7 | 3/7 |
| All human files flagged | 6/14 | 6/14 | 3/14 | 6/14 |
| Human source groups with a flag | 4/7 | 4/7 | 2/7 | 3/7 |
| Qwen rewrites usefully localized | 1/4 | 1/4 | 2/4 | 2/4 |
| SmolLM2 rewrites usefully localized | 3/5 | 2/5 | 3/5 | 2/5 |
| All rewrites usefully localized | 4/9 | 3/9 | 5/9 | 4/9 |
| AI files flagged only at a wrong location | 0/9 | 2/9 | 1/9 | 1/9 |

The matched fit retained all four useful current locations, recovered one more and removed three human flags without adding any. The two-sided fit gained one useful Qwen location and lost one SmolLM2 location. It removed three human flags and added three others. Both fits put the selected region on the exact sentence. The two-sided arm failed its advancement criteria because its useful count was unchanged against the current context and lower than the matched fit.

Across the two small technical-publisher panels, both fitted arms found **7 of 15** rewrites. The matched full-window fit flagged **5 of 26** human files and the two-sided fit flagged **11 of 26**. The current check found **5 of 15** and flagged **12 of 26**. These are descriptive totals over designed cases, with correlated variants within each author group. The matched fit is the stronger candidate for a broader evaluation with additional publishers and editing styles.

The local classifier executed 18 forwards for 143 unique complete inputs. A separate local replay checked all 23 decoded TXT fixtures, all 14 original and donor source locations, nine exact rewrite foci, 276 context geometries, 120 outcome counts and four human-group counts. The public aggregate includes commitments and measured outcomes. Raw source text and generated replacements stay in the local trial directory.

## Local app verification

Enable `VERITAS_TEXT_MATCHED_CONTEXT_REVIEW=true` in a development instance running the candidate text check. The rule covers mapped English plain text with one paragraph, four sentences and 68–134 words. A selected sentence must have at least 12 words and an entirely measured three-sentence context. The rule reuses the existing response and carries zero weight in the whole-document assessment. Other passage checks retain their own responses. This mode is disabled by default and is limited to development instances.

An isolated Full-profile smoke used two already evaluated versions of one USGS passage, with the feature off and on. The check recovered the known edited sentence at extracted code points `[133, 305)` and selected neither eligible sentence in the human original. The analysis decisions and underlying model scores matched with the feature off and on. The flow queued zero analyses before a valid simulated Stripe payment. API evidence, JSON export and the signed package matched, and the package verifier reproduced the source positions and exact context commitments from the original file. This smoke checks the serving path on known inputs. The two-panel comparison above measures detection performance.

## Reproduce the holdout comparison

The [transfer runner](../backend/scripts/writing_two_sided_transfer.py) accepts the locally prepared `source.json`, `private-holdout.jsonl`, `generation-freeze.json` and `admission.json`. The holdout uses the [prior runner's row format](writing-two-sided-context-2026-11-11.md). Supply source-author lists in the new roster. The runner checks the exact published fit and method before freezing the comparison and uses the existing pinned local classifier.

```sh
python -m scripts.writing_two_sided_transfer freeze RUN_DIR
python -m scripts.writing_two_sided_transfer run RUN_DIR --snapshot LOCAL_MODEL
```
