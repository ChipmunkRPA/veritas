# Passage evidence on practical notices

This local transfer used agency-authored procedural prose from four official 2019–2021 Federal Register issues. Six of 27 planned notices qualified, with one notice per issuing department or independent agency. Eighteen did not meet the continuous-prose and matched-donor requirements. Two repeated an admitted department and one shared a normalized 12-token fragment with another eligible notice. The selected prose excluded contact fields, marked quotations, third-party submissions identified by the intake rules, and footnotes. The admitted human cases shared no exact normalized 12-token fragments with the 45,047 entries in the 34 checked historical packets. Model pretraining exposure is unknown. [Official Federal Register collection](https://www.govinfo.gov/help/fr) · [U.S. government works provision](https://www.copyright.gov/title17/92chap1.html)

Each of the six notices supplied an original excerpt, a complete human paragraph replacement, and a two-sentence human replacement. Both replacements came from length-matched donor prose elsewhere in the same agency notice. Whole replacements spanned 74–129 words and two-sentence replacements spanned 38–67 words. The controls are constructed edits made from known human-origin text.

Two Apache-2.0 generator families each attempted one target-only paragraph rewrite per notice offline. All 12 responses ended naturally. Eight qualified, four per family. Four copied too much source wording under the unchanged admission rule. The admitted rewrites cover all six agencies, and two sources have both families represented. Exact surrounding human text and Unicode positions were verified.

Every admitted case was saved as a DOCX with two text runs per body paragraph and decoded through the app. All 26 files returned the identical text, paragraph and run mappings, and full source coverage. The classifier comparison used those verified text inputs. The reference uses the current local classifier’s raw sentence-context review rule. The comparison also reuses the prior strict point and frozen isolated paragraph and neighbor rules. The notice panel supplied no fitting inputs. A useful location requires at least 0.30 non-whitespace character intersection over union with the known generated paragraph. [Frozen comparison method](writing-paragraph-context-transfer-2026-11-11.md) · [Exact outcomes and source receipts](evaluation-panels/writing-practical-notice-transfer-2026-11-11.json)

## Results

| Outcome | Current raw review | Strict raw | Isolated paragraph | Two-sided contrast | Both paragraph checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Human originals flagged | 2/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| Human paragraph replacements flagged | 2/6 | 1/6 | 0/6 | 0/6 | 0/6 |
| Human two-sentence replacements flagged | 1/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| All human controls flagged | 5/18 | 1/18 | 0/18 | 0/18 | 0/18 |
| Qwen rewrites usefully localized | 3/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| SmolLM rewrites usefully localized | 3/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| All eligible rewrites usefully localized | 6/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| Rewrites without a useful location | 2/8 | 8/8 | 8/8 | 8/8 | 8/8 |

The CPU comparison made 156 classifier forwards. All 26 admitted cases had full scored coverage under every rule. The current raw review found useful locations in both generator families and also marked five human files. Its mark overlapped a genuine editorial replacement in one file. The stricter rule marked one genuine replacement and found zero generated locations. None of the three frozen paragraph rules made a mark on this genre. The four source-copy exclusions remain in the preparation denominator.

The practical-documents result supports keeping the current raw review for inspection. Its five human flags show the limits of treating a marked passage as proof of authorship. The sample has six agencies, two generator families, and one kind of ordinary document. The study leaves customer verdicts and confidence calibration outside its scope. [Fiction transfer](writing-fiction-paragraph-transfer-2026-11-11.md) · [Execution and exact anchors](evaluation-panels/writing-practical-notice-transfer-2026-11-11.json)
