# Passage review on archived email bulletins

Six U.S. federal publishers provided dated English-language email-bulletin prose from 2021. Each group came from a different department or independent agency. The excerpts use official prose and publisher-authored labels or sign-offs. Vendor boilerplate, subscriber details, contact information and third-party quotations were excluded. All six groups passed an exact normalized 12-token overlap screen against 45,113 retained project text entries. Model-pretraining exposure remains unknown. [U.S. government works](https://www.copyright.gov/title17/92chap1.html) · [EPA](https://content.govdelivery.com/accounts/USEPAORD/bulletins/2d67d90) · [Agriculture](https://content.govdelivery.com/accounts/USFSIS/bulletins/2e868bd) · [Homeland Security](https://content.govdelivery.com/accounts/USDHSFEMA/bulletins/2fe8d83) · [Commerce](https://content.govdelivery.com/accounts/USDOC/bulletins/2f7a66f) · [Labor](https://content.govdelivery.com/accounts/USDOL/bulletins/2fef2c2) · [Treasury](https://content.govdelivery.com/accounts/USTREAS/bulletins/2f04eeb)

Each bulletin supplied an original excerpt, a whole-paragraph replacement, a one-sentence replacement, an excerpt with a marked quote, and an excerpt with a publisher label or sign-off. The replacements and additions use exact prose from the same issue. These are constructed editing controls. For each source, two Apache-2.0 model families attempted one target-paragraph rewrite apiece. Eight of twelve passed the frozen source-copy and other eligibility checks, with three Qwen and five SmolLM rewrites. Four outputs copied too much source wording. The eight admitted rewrites covered five source groups. Each was measured plain and again with exactly the same added quote and template. [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

The app extracted the exact selected UTF-8 plain body from 46 synthetic multipart EML files and preserved the character ranges and quote/signature roles. An alternate HTML part and dummy attachment stayed outside the selected body. The original correspondence content is archived, and all evaluated EML envelopes were constructed. This panel does not represent private inbox traffic, authentic reply chains, or sender authentication.

## Held-out results

The current local raw review, previous strict point, pooled point, and practical point were fixed before generation or scoring. Useful localization required at least 0.30 non-whitespace character intersection over union with the known edited paragraph. The detector made 150 CPU forwards and all 46 eligible files had complete scored text coverage. [Prior practical fit](writing-correspondence-transfer-2026-11-11.md) · [Frozen aggregate and source commitments](evaluation-panels/writing-modern-bulletin-transfer-2026-11-11.json)

| Outcome | Current raw review | Strict raw and pooled | Practical point | Practical point, body-only contexts |
| --- | ---: | ---: | ---: | ---: |
| Original controls flagged | 2/6 | 1/6 | 2/6 | 2/6 |
| Human paragraph replacements flagged | 2/6 | 1/6 | 2/6 | 2/6 |
| Human sentence replacements flagged | 2/6 | 1/6 | 2/6 | 2/6 |
| Human quote controls flagged | 2/6 | 1/6 | 2/6 | 2/6 |
| Human template controls flagged | 2/6 | 1/6 | 2/6 | 2/6 |
| All human controls flagged | 10/30 | 5/30 | 10/30 | 10/30 |
| Human source groups with a flag | 2/6 | 1/6 | 2/6 | 2/6 |
| Plain Qwen rewrites usefully localized | 3/3 | 0/3 | 2/3 | 2/3 |
| Plain SmolLM rewrites usefully localized | 5/5 | 2/5 | 4/5 | 4/5 |
| Same Qwen rewrites with quote and template | 3/3 | 0/3 | 2/3 | 2/3 |
| Same SmolLM rewrites with quote and template | 5/5 | 2/5 | 4/5 | 4/5 |

The practical point missed two rewrites found by current raw review and kept the same human flags. Strict and pooled review each cleared five human flags and missed six rewrites. The five human views in a source group share text, so the group counts are the more meaningful measure of false alarms in this panel.

Appending the quote or template left every human file-level flag unchanged. Appending both to the same rewrites also left useful localization unchanged. The practical point still selected contexts intersecting quote or signature material on one human quote control and four decorated mixed files. Requiring the entire recorded three-sentence context to stay inside the message body removed those selections without changing any human file-level flag or useful localization. All 46 files retained body-only contexts. These six groups support showing whether recorded evidence crosses email roles. They do not support changing the serving detector or overall confidence.
