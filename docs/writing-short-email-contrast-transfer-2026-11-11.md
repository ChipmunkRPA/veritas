# Short workplace email: contrasting sentences and omission inputs

The earlier short email test found that isolated sentence scoring located four of five admitted rewrites and flagged three of eight human examples. This transfer asks whether contrasting a sentence with the other sentences in its message, or measuring the response when that sentence is omitted, can retain precise locations with fewer false flags. LA-CPD describes how topic and local content variation can shift detector responses within a single author's writing. This comparison uses its own bounded rules for single rewrites in short messages. It did not import the paper’s code, weights or dataset. [Earlier short email comparison](writing-short-email-comparison-2026-11-11.md) · [LA-CPD study](https://arxiv.org/html/2609.33787v1)

## Sources and fixed comparison

Six new writer and message groups came from two public EPA releases. Each group contains a 45–94-word government-duty email excerpt and one genuine source sentence substituted for the exact target sentence. Five donor sentences were by the same writer, including two from outside the retained excerpt in the same source message. One donor was by a documented federal colleague in the same workplace thread. The groups cover planning, announcements, science review, a visit, fisheries consultation and economic-model discussion. Some subjects are institutionally related. Contacts, signatures, private third-party passages and redactions were outside the scored text. The originals and donor controls had no exact normalized 12-token overlap with another group or 45,260 known prior project text entries. Model pretraining exposure is unknown. [EPA Release 4](https://19january2017snapshot.epa.gov/sites/production/files/2014-01/documents/release-4-part-e.pdf) · [EPA correspondence release](https://www.epa.gov/foia/401-info-e-mail-correspondence) · [U.S. government works](https://www.copyright.gov/title17/92chap1.html)

Two pinned Apache-2.0 generator families attempted one exact sentence rewrite per group. The same completion, length and copy rules as the earlier test admitted six of six Qwen and three of six SmolLM responses. The other three SmolLM attempts failed length, sentence structure and direct-copy rules. Each admitted edit preserved all text outside the target. Synthetic multipart EML fixtures preserved the exact decoded text/plain source positions. The alternate HTML and a fixture attachment were outside the measured body. [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

- **Current passage** uses the current three-sentence contexts and the first six positive foci. The union of the contexts determines measured location.
- **Candidate sentence** uses the previously tested Apache-2.0 binary classifier on each exact sentence of at least 12 words. **Sentence and body** also requires the same classifier to flag the intact decoded body. Both use the fixed candidate reference from the prior test.
- **Relative sentence** requires a candidate sentence flag and twice the odds of the strongest other eligible sentence in that message.
- **Matched omission** requires a candidate sentence flag, a decrease by a factor of at least two in the whole-body response odds after omitting that sentence, and a larger decrease than after omitting any other eligible sentence between half and twice its length. The reduced input retains all other sentences in source order with a single space between them. **Relative and omission** requires both rules.

A candidate can mark only the exact sentence it scored. Omitting one sentence changes length and transitions. Its response measures sensitivity to that edit and cannot by itself attribute authorship. Useful localization required at least 0.30 non-whitespace character intersection over union with the known edit. Methods were frozen before generation and no points were fitted on these outcomes. [Candidate model and license](https://huggingface.co/rasbt/ai-text-detector-modernbert)

## Local results

All 21 admitted EML files passed exact source binding. Two CPU scorers made 73 forward calls, covering 73 current context windows, 21 candidate body inputs, 64 isolated sentences and 64 omission inputs. Every candidate input had one complete token window. The independent arithmetic readout reproduced 96 outcome counts. [Aggregate results](evaluation-panels/writing-short-email-contrast-transfer-2026-11-11.json)

| Outcome | Current passage | Candidate sentence | Sentence and body | Relative sentence | Matched omission | Relative and omission |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Human originals flagged | 0/6 | 4/6 | 1/6 | 3/6 | 0/6 | 0/6 |
| Genuine human sentence edits flagged | 0/6 | 3/6 | 1/6 | 2/6 | 0/6 | 0/6 |
| All human cases flagged | 0/12 | 7/12 | 2/12 | 5/12 | 0/12 | 0/12 |
| Human source groups with a flag | 0/6 | 4/6 | 1/6 | 4/6 | 0/6 | 0/6 |
| Qwen edits usefully localized | 1/6 | 5/6 | 1/6 | 3/6 | 1/5 | 0/5 |
| SmolLM edits usefully localized | 2/3 | 3/3 | 1/3 | 2/3 | 0/3 | 0/3 |
| All rewrites usefully localized | 3/9 | 8/9 | 2/9 | 5/9 | 1/8 | 0/8 |
| Admitted rewrites unassessed by rule | 0 | 0 | 0 | 0 | 1 | 1 |

The sentence-only check retained all three current useful locations and found five more. It also highlighted the human replacement in all three flagged donor controls. Relative sentence contrast found four locations the current check missed, lost two current locations and marked only the wrong sentence in two AI-edited files. Whole-body corroboration missed two of the three locations found by the current check and found one additional location. Matched omission retained no current useful location and found one new location. Both contrast rules together marked no file.

The panel tests short English government work emails and controlled single-sentence replacements from two generator families. It does not estimate real-world prevalence or validate commercial correspondence generally. The source groups are small, one model supplies the new candidate responses at every scope, and results are sensitive to the generator eligibility rules. The new rules remain research-only. Production passage selection and confidence are unchanged.
