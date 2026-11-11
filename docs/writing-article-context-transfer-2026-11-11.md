# Passage review with wider context and paragraph omission

The confirmation panel used six dated English-language web features from NASA Earth Observatory, NIST, NOAA Fisheries, NOAA NESDIS and USDA Agricultural Research Service. The text came from the publishers’ body prose. Captions, navigation, external quotations and linked third-party content were excluded. Each article supplied a four-paragraph original excerpt, a length-matched paragraph substitution and a one-sentence substitution drawn from elsewhere in the same article. The selected originals and ordinary controls shared no normalized 12-token fragments with 45,159 retained project text entries or with any other confirmation article. Model-pretraining exposure is unknown. [U.S. government works](https://www.copyright.gov/title17/92chap1.html) · [NASA Earth Observatory reuse policy](https://science.nasa.gov/earth/faq/) · [NOAA copyright terms](https://repository.library.noaa.gov/Content%20and%20Copyright)

Each article had one interior paragraph rewritten by each of two Apache-2.0 generator families. Eight of twelve responses passed the fixed eligibility checks. Three Qwen and five SmolLM rewrites qualified, and the other four copied too much of the original. Every admitted rewrite retained the exact surrounding paragraphs. These are controlled edits in a small sample of federal features. The NIST page lists a later update, and publishers and topics are clustered. [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

## Frozen comparison

The current review and the earlier practical, strict and pooled operating points retained their usual three-sentence inputs and the first-six selection cap. The new variants considered either interior paragraph. The paragraph-only rule selected the paragraph containing the strongest qualifying current response. The next two variants required a score of at least 0.5 on the paragraph alone and then on the paragraph plus both neighbors. The final variant also required the wider score to drop when the target was removed and required that drop to exceed the drop when removing either neighbor. These comparisons share classifier weights. Removing text changes length, topic and transitions, so the change measures the classifier’s sensitivity to that input. [Prior isolated-paragraph experiment](writing-paragraph-context-transfer-2026-11-11.md)

The evaluation pairs each original with controlled revisions, following the matched evaluation design in ARB. LA-CPD motivates checking whether apparent local changes survive nearby evidence, since content differences alone can vary detector responses. This study implements its own omission comparison using existing local weights. No paper code, weights or datasets were used. Useful localization required at least 0.30 non-whitespace character intersection over union with the exact rewritten paragraph. [ARB](https://arxiv.org/html/2607.29539v1) · [LA-CPD](https://arxiv.org/html/2609.33787v1)

## Held-out results

The classifier performed 161 CPU forward calls on 26 eligible files. The other four planned files were excluded by the generator eligibility checks before scoring. Every current and comparison input was checked against the model’s actual tokens. A separate bitset replay matched 25 arm-by-rule counts and five human group counts. [Frozen aggregate and source commitments](evaluation-panels/writing-article-context-transfer-2026-11-11.json)

| Outcome | Current raw | Practical | Strict raw | Pooled | One paragraph | Isolated confirmation | Wider confirmation | Paragraph omission |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Human originals flagged | 5/6 | 3/6 | 2/6 | 1/6 | 3/6 | 1/6 | 0/6 | 0/6 |
| Human paragraph replacements flagged | 5/6 | 4/6 | 2/6 | 1/6 | 1/6 | 0/6 | 0/6 | 0/6 |
| Human sentence replacements flagged | 5/6 | 3/6 | 2/6 | 1/6 | 2/6 | 1/6 | 0/6 | 0/6 |
| All human files flagged | 15/18 | 10/18 | 6/18 | 3/18 | 6/18 | 2/18 | 0/18 | 0/18 |
| Human source groups with a flag | 5/6 | 4/6 | 2/6 | 1/6 | 4/6 | 1/6 | 0/6 | 0/6 |
| Qwen rewrites usefully localized | 3/3 | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 | 2/3 | 1/3 |
| SmolLM rewrites usefully localized | 5/5 | 4/5 | 2/5 | 2/5 | 5/5 | 5/5 | 3/5 | 3/5 |
| All rewrites usefully localized | 8/8 | 7/8 | 4/8 | 4/8 | 8/8 | 8/8 | 5/8 | 4/8 |

Requiring the isolated paragraph response retained every useful location from current raw review, removed 13 human file flags, and introduced none. Selecting one paragraph without that additional forward removed nine human file flags and also retained every location. Wider context removed the two remaining human flags while missing three rewrites. The omission rule cleared no additional false flags and missed one more rewrite.

## Local opt-in

Set `VERITAS_TEXT_ISOLATED_PARAGRAPH_REVIEW=true` in the local backend environment to show the extra reading for eligible plain-text analyses. It examines at most two bounded interior paragraphs and preserves the context and paragraph input hashes and offsets. The option defaults to off. The extra response adds inspection context in the app, PDF, and signed JSON. The evaluated flag-reduction rule remains a study result, and the app uses its existing finding selection and overall confidence.

The isolated-paragraph check merits a new prospective test in other writing genres. This panel contains only six source groups and synthetic single-paragraph edits, and both responses share classifier weights. The app’s detector and confidence calculation remain unchanged.
