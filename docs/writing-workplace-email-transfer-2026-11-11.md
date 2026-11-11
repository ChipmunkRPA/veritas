# Isolated paragraph review on archived workplace email

This panel uses eleven messages selected from 17 National Archives release packets of records-management correspondence between U.S. government employees. The messages were written between 2009 and 2015 and address ordinary work such as project rollout, meeting coordination and requests to review records procedures. Only clearly attributable government-duty prose entered the fixtures. Contact details, headers, attachments, quoted reply material and signatures were excluded. The original archive contains PDF copies and some text-layer OCR artifacts. The committed source locations identify the exact extracted paragraphs. The human body excerpts and controls shared no exact normalized 12-token fragment with 45,220 retained project text entries or with another admitted source. Training exposure remains unknown. [National Archives release](https://www.archives.gov/foia/state-department-emails) · [U.S. government works](https://www.copyright.gov/title17/92chap1.html)

Four messages from different employees supplied four-paragraph excerpts of 196 to 328 words. Each long source contributed its original excerpt, a paragraph replacement and a sentence replacement drawn from exact prose by the same employee. Two Apache-2.0 generator families attempted one rewrite each of the same interior paragraph. Seven different employees supplied shorter original bodies of 20 to 118 words in one to three paragraphs. They are counted separately because the existing paragraph rule requires at least four source paragraphs. No short mixed cases were constructed. [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

Each fixture wrapped the selected text in a synthetic multipart EML. The app had to choose the exact plain-text body, preserve its Unicode positions, and exclude the alternate HTML part and the fixture attachment. This setup does not authenticate archive senders or exercise mixed quoted content. The production paragraph option remains scoped to plain text. The research run projected the exact selected body into that same scorer, with fixed eligibility and raw response rules, after confirming that the production scorer declined the email document. Useful localization required at least 0.30 non-whitespace character intersection over union with the known replacement. [Earlier paragraph replication](writing-paragraph-replication-2026-11-11.md)

## Results

All 27 eligible EML files passed the exact source-map checks. The classifier made 115 CPU forward calls. All eight rewrites passed the frozen preparation rules, including one recovered with the prespecified complete-sentence prefix. Two of 40 interior paragraphs qualified for additional paragraph-only scoring. Both were read with complete single-window coverage. A separate readout matched 18 outcome-and-transition counts, six source-group counts and all recorded EML and paragraph hashes. [Aggregate and source commitments](evaluation-panels/writing-workplace-email-transfer-2026-11-11.json)

| Outcome | Current context | Single paragraph | Single paragraph with isolated response |
| --- | ---: | ---: | ---: |
| Long original files flagged | 1/4 | 0/4 | 0/4 |
| Ordinary whole-paragraph edits flagged | 1/4 | 0/4 | 0/4 |
| Ordinary sentence edits flagged | 1/4 | 0/4 | 0/4 |
| Long human files flagged | 3/12 | 0/12 | 0/12 |
| Long human source groups with a flag | 1/4 | 0/4 | 0/4 |
| Qwen rewrites usefully localized | 2/4 | 2/4 | 2/4 |
| SmolLM rewrites usefully localized | 0/4 | 0/4 | 0/4 |
| All long rewrites usefully localized | 2/8 | 2/8 | 2/8 |
| Short human bodies flagged | 0/7 | Unassessed | Unassessed |

All three long human flags came from variants of the same message. Narrowing to an interior paragraph removed those flags and retained both useful locations. The isolated response made no further selection change. The current review also flagged one SmolLM rewrite outside the known edited paragraph. The older stricter rules suppressed all human flags and localized none of the eight rewrites.

Six of eight mixed edits were missed across the main comparisons. These four eligible source groups and the lack of short mixed cases leave workplace-email effectiveness unresolved. The app’s email scope and confidence calculation remain unchanged.
