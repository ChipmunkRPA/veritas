# Frozen short-message focus selection on NASA correspondence

The earlier Department of Justice holdout found that a source-calibrated window rule localized two of five rewrites the current check missed, with the same two human false flags. This transfer tests that frozen rule on a new publisher and work domain. It also separates changes in what gets selected from changes in how broadly it is marked. [Earlier held-out comparison and fitted points](writing-context-token-transfer-2026-11-11.md)

## Sources and methods

Five attributable NASA employees supplied distinct mission-work messages published in the 2003 Columbia Accident Investigation Board report. The 56–68-word excerpts concern scheduling, internal review, engineering status, media briefing, and a technical exchange. All came from one agency and investigation, so author and thread separation does not eliminate shared institutional language. Each group has one exact donor sentence from the same writer. Some donors are omitted text from the same original message. The six previously consumed fit groups, four cutoff groups and four DOJ confirmation groups were never reused as test inputs. No exact normalized 12-token fragment matched another admitted group or 45,294 retained prior project entries. Model-pretraining exposure remains unknown. Only clearly attributable federal-duty passages were scored. [NASA report](https://www.nasa.gov/wp-content/uploads/static/history/columbia/reports/CAIBreportv1.pdf) · [U.S. government works](https://www.copyright.gov/title17/92chap1.html)

The two pinned Apache-2.0 generators each attempted one exact target-sentence rewrite per group under the earlier completion, length and copy rules. All admitted files preserve the exact text outside the target. Each synthetic multipart EML was decoded through the app to its exact plain-text body, with the alternate HTML and test attachment excluded. The recorded focus and context ranges point into that decoded body. Mail headers in the test fixtures are neutral and do not establish sender authenticity. [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

The same pinned Apache-2.0 classifier scored the focus sentence and its neighbors as one input. The two prior fitted review cutoffs and their minimum 12-word focus requirement were reused unchanged. Each cutoff has a full-context and a focus-only display comparison with identical selected foci. The current raw rule likewise has both display comparisons, retaining all of its qualifying foci. Useful localization required at least 0.30 non-whitespace character intersection over union with the known edit. A focus-only range identifies where to read first. The response was measured on the whole recorded input. [Pinned classifier](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

## Results

Four of five Qwen rewrites passed the fixed generation checks. None of the five initial SmolLM attempts passed. Three produced multiple sentences and two copied too much. A separate SmolLM pass was frozen after those generation exclusions and before opening any detector outcomes. It used one revised prompt and seed on all five sources with the original admission rules. Two rewrites passed and three copied too much. The two families therefore contributed four and two admitted rewrites, respectively.

The first run checked 14 EML fixtures and the second checked 12, including the same ten human controls in each. The controls reproduced identical selected source geometry. All six admitted rewrite foci passed exact source and token binding. The two runs completed 28 native forward calls. An independent readout reproduced the source ranges and 72 outcome counts in each run. [Aggregate by family and frozen cutoff](evaluation-panels/writing-focus-publisher-transfer-2026-11-11.json)

| Outcome | Current context | Current focus only | Stricter fit, context | Stricter fit, focus | More permissive fit, context | More permissive fit, focus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Human originals flagged | 0/5 | 0/5 | 2/5 | 2/5 | 2/5 | 2/5 |
| Genuine human sentence edits flagged | 0/5 | 0/5 | 1/5 | 1/5 | 1/5 | 1/5 |
| Human source groups with a flag | 0/5 | 0/5 | 2/5 | 2/5 | 2/5 | 2/5 |
| Qwen rewrites usefully localized | 2/4 | 1/4 | 3/4 | 3/4 | 3/4 | 3/4 |
| SmolLM rewrites usefully localized | 0/2 | 0/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| All rewrites usefully localized | 2/6 | 1/6 | 4/6 | 4/6 | 4/6 | 4/6 |
| AI rewrites flagged only at a wrong location | 0/6 | 1/6 | 0/6 | 0/6 | 0/6 | 0/6 |

The frozen rule added two useful locations across the two generator families and flagged three human files across two source groups. The earlier DOJ gain at unchanged false alarms did not carry over. At the current operating point, limiting the highlight to the focus sentence lost one useful location because the model's actual context overlapped the edit while its navigation sentence did not. At the fitted points, the two display scopes produced the same useful and false-flag counts on these files. The new rules remain experimental. Production selection and confidence stay unchanged.
