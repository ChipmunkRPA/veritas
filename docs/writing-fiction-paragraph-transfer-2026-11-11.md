# Paragraph evidence on historical fiction

This local transfer used a genre outside the earlier science-explainer panel. We checked 12 English fiction works from distinct authors, with each catalog record identifying the work as public domain in the United States. Eleven qualified, and one had a normalized 12-token match in the earlier project inventory. The check covered 44,992 text entries in 32 packets. We removed the digitization wrapper and selected prose from the middle of each book. Pretraining exposure is unknown. [Publisher license policy](https://www.gutenberg.org/policy/license.html) · [Previous paragraph comparison](writing-paragraph-context-transfer-2026-11-11.md)

Each work supplies an unchanged excerpt, a full human paragraph replacement, and a two-sentence replacement inside the original paragraph. The replacements are authentic prose taken from elsewhere in the same book, with donor lengths within 25% of the removed material. The full replacements span 55–105 words, with a median of 92. The two-sentence replacements span 21–65 words, with a median of 51. These are constructed human-origin controls. Every variant preserves exact Unicode offsets and the surrounding original text.

Both Apache-2.0 generator families used the same fiction-specific prompt and made one offline rewrite attempt per work. All 22 mixed variants were admitted. Twenty-one used the complete response. One used the longest complete-sentence prefix under the prospectively fixed length recovery. All retain the same human text outside the replaced paragraph. The generator attempt dispositions, licenses and source attribution appear in the aggregate. [Exact results](evaluation-panels/writing-fiction-paragraph-transfer-2026-11-11.json)

The classifier, sentence-window baseline, operating points and combined paragraph check were frozen on the earlier development sources. The holdout supplied no fit inputs. The paragraph check requires both a high isolated response and a rise in log-odds above the stronger adjacent paragraph. It marks at most one unique top qualifying paragraph. The metric requires at least 0.30 non-whitespace character intersection over union with the known generated span. Model input tokens and shown source extents were verified. [Frozen point and geometry](writing-paragraph-context-transfer-2026-11-11.md)

## Results

| Outcome | Raw review | Strict raw | Isolated paragraph | Two-sided contrast | Both paragraph checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Human originals flagged | 6/11 | 0/11 | 0/11 | 1/11 | 0/11 |
| Full human paragraph replacements flagged | 5/11 | 0/11 | 0/11 | 1/11 | 0/11 |
| Two-sentence human replacements flagged | 5/11 | 0/11 | 0/11 | 1/11 | 0/11 |
| All human controls flagged | 16/33 | 0/33 | 0/33 | 3/33 | 0/33 |
| Qwen rewrites usefully localized | 10/11 | 4/11 | 3/11 | 5/11 | 2/11 |
| SmolLM rewrites usefully localized | 10/11 | 9/11 | 9/11 | 9/11 | 8/11 |
| All eligible rewrites usefully localized | 20/22 | 13/22 | 12/22 | 14/22 | 10/22 |
| Eligible rewrites without a useful location | 2/22 | 9/22 | 10/22 | 8/22 | 12/22 |

All 55 files were assessed. The CPU run made 367 classifier forwards. The raw review highlighted the authentic edited area on two whole-paragraph replacements and one two-sentence replacement. The two-sided rule alone added one useful location relative to strict raw and flagged three additional human files. Requiring both paragraph checks kept ten of the strict rule's useful locations and lost three. Strict raw and the combined rule each flagged zero human controls.

On science explainers, the combined rule localized six of 18 edits and the strict rule localized four. On fiction, the combined rule localized ten of 22 and the strict rule localized 13. The genre transfer does not support replacing the stricter reference. The paired controls are constructed from same-author prose, and the two generator families produce a limited set of edit styles. This evaluation adds no customer verdict rule or confidence calibration. [Science-explainer transfer](writing-paragraph-context-transfer-2026-11-11.md) · [Fiction execution receipts](evaluation-panels/writing-fiction-paragraph-transfer-2026-11-11.json)
