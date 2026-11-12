# Isolated paragraph review on transcribed human speech

Choosing one paragraph and requiring a second reading of it reduced human-file highlights from 24 of 27 to 12 of 27 and still localized the same 12 rewrites. Four of six oral-history speakers still had a human file highlighted. [Audited aggregate](evaluation-panels/writing-speech-paragraph-transfer-2026-11-11.json)

## Sources and freeze

The panel comprises six National Archives oral histories and three archived White House public addresses, with a distinct speaker per source. It extends the earlier prepared Federal Reserve speech panel to retrospective interviews and separately published addresses. Intake used the native PDF text layer for interviews and the speech bodies from archived HTML. It kept attributed blocks from the selected speaker and removed speaker labels and stage directions. The four-block fixtures may include different turns from the same speaker. Their presentation differs from a verbatim transcript containing speaker labels or interviewer questions. [National Archives interview rights](https://www.archives.gov/presidential-libraries/research/transition-interviews/copy-rights.html) · [Archived White House publications](https://www.archives.gov/presidential-records/research/archived-white-house-websites) · [Earlier court and prepared-speech transfer](writing-paragraph-replication-2026-11-11.md)

Each group supplied an original, a same-speaker paragraph splice, and a same-speaker sentence splice. The selected human texts shared no normalized exact 12-token fragment with any of the 45,408 entries in 53 retained project packets or with another held group. The classifier's pretraining exposure remains unknown. Both generator families each attempted one rewrite of the same interior paragraph. Sixteen of 18 attempts passed the unchanged length, copy, format and source-binding checks. Each retained rewrite preserved the surrounding source text. Both families contributed eight admitted rewrites. [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

The frozen comparison reused the production three-sentence classifier inputs and earlier isolated-paragraph rule. The current review displays the first six source-ordered qualifying contexts. The one-paragraph arm selects the eligible interior paragraph with the strongest context response. Isolated review requires an additional qualifying reading from that whole paragraph, with the same classifier weights and complete token coverage. A useful location needs at least 0.30 intersection over union with the known rewrite, measured in non-whitespace characters. No point was fitted on these sources.

## Local results

| Outcome | Current context | Single paragraph | Isolated review |
| --- | ---: | ---: | ---: |
| Oral-history human files flagged | 18/18 | 14/18 | 9/18 |
| Oral-history human groups with a flag | 6/6 | 6/6 | 4/6 |
| Oral-history rewrites usefully localized | 9/11 | 9/11 | 9/11 |
| Public-address human files flagged | 6/9 | 6/9 | 3/9 |
| Public-address human groups with a flag | 2/3 | 2/3 | 1/3 |
| Public-address rewrites usefully localized | 3/5 | 3/5 | 3/5 |
| All human files flagged | 24/27 | 20/27 | 12/27 |
| All rewrites usefully localized | 12/16 | 12/16 | 12/16 |

All three arms localized the same 12 rewrites. The single-paragraph cap alone removed four human flags. The isolated reading removed eight more and introduced none. A stricter context point flagged zero human files and localized nine rewrites. Both genres draw from a small set of U.S. institutions, and versions from a speaker are correlated. The isolated reading supports optional evidence prioritization. Its remaining human highlights and the shared detector weights leave it outside authorship confidence and automatic verdicts.

The CPU run scored 43 of 45 planned files and made 274 forwards. The remaining two generator responses failed admission before classification. A separate model-free verifier matched nine original publisher files, 43 scored inputs, 51 complete paragraph reads, and 30 primary rollups and paired transitions. It also recalculated source-group totals because the evaluation readout retained the previous study's genre labels for one summary field. The published packet records that correction and the publisher's corrected date for the Pence address. The source bytes, frozen responses and case-level evidence remain local. [Independent verifier](../backend/scripts/review_speech_paragraph_transfer.py)

From the repository root, replay the published summary using a directory with the retained private intake, raw publisher files, prepared cases and scored receipts.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_speech_paragraph_transfer \
  "$run_dir" docs/evaluation-panels/writing-speech-paragraph-transfer-2026-11-11.json
```
