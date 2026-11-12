# Direct sentence judgments under all six answer orders

The earlier copyediting probe recovered some known rewrites and flagged too much genuine prose. This study instead asks two small local instruction models which of two interior sentences, if either, shows stronger evidence of an AI rewrite. Each model sees the same source in the same sentence order. All six assignments of the answer letters are tested. Research documents that the same multiple-choice input can receive different judgments after its answers are reordered. [Earlier copyediting test](writing-blind-rewrite-probe-2026-11-11.md) · [Option-order sensitivity research](https://aclanthology.org/2024.findings-naacl.130/) · [Recent permutation scoring research](https://aclanthology.org/2026.findings-emnlp.1290/)

## Publisher sources and scoring

The test used dated 2019–2020 releases from the Federal Aviation Administration and USDA Farm Service Agency. Each admitted release supplied a four-sentence excerpt, a human control using a separate length-matched sentence from the same release, and one controlled rewrite attempt per local generator family. The models scored each completed version without seeing an earlier draft, the generator identity or the selected target. Source preparation excluded quotes, contact information and image captions. Both generator checkpoints are Apache-2.0. [Rights guidance for federal material](https://www.usa.gov/government-copyright) · [USDA archival rights guidance](https://www.nal.usda.gov/programs/speccoll) · [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM2 license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/LICENSE)

A restart cleared the first intake and its local scoring files. The rerun used the same twelve publisher URLs and a new freeze. The recovery screened each excerpt and donor against 38,910 English text entries from twelve retained benchmark packets and found no shared normalized 12-word fragments in the admitted material. Earlier temporary benchmark packets were unavailable. The recovered intake admitted all twelve sources, including one FSA release excluded in the interrupted intake. [Source preparation](../backend/scripts/prepare_writing_option_order_transfer.py)

For each order, the study captured native next-token probability mass for three exact answer letters. The first-order comparison used one assignment. The balanced comparison required all six orders to produce at least 0.20 total mass over allowed answer tokens. It accepted a semantic answer only when it led in at least four orders and its geometric mean answer score was at least 1.5 times the next highest. The two-model agreement arm required both models to pick the same source sentence. These numbers are operating rules for the experiment. Answer-token probabilities do not estimate real-world AI authorship.

The serving comparison used the existing first six positive three-sentence classifier windows. A useful proposal had at least 0.30 non-whitespace character overlap by intersection over union with the known replacement. The fixed advance rule required the union to add useful locations in both publishers and both generator families with at most one additional human source group flagged. This is a prospective two-publisher development comparison. The models that generated rewrites also supplied the individual judgments, so the results distinguish the edit families and the study reports limited transfer. [Research runner](../backend/scripts/writing_option_order_transfer.py) · [Independent replay](../backend/scripts/review_writing_option_order_transfer.py)

## Local results

Nineteen of 24 generated rewrites met the frozen requirements for a distinct, complete replacement sentence. The 43 scored files include 24 human controls. A positive result can select the wrong interior sentence, so the table counts useful locations against the exact known edit.

| Outcome | Current context | Qwen fixed | Qwen all orders | SmolLM2 fixed | SmolLM2 all orders | Agreement | Current plus agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Useful locations on Qwen edits | 1/8 | 1/8 | 0/8 | 0/8 | 0/8 | 0/8 | 1/8 |
| Useful locations on SmolLM2 edits | 1/11 | 2/11 | 0/11 | 0/11 | 0/11 | 0/11 | 1/11 |
| Untouched human files flagged | 3/12 | 4/12 | 2/12 | 0/12 | 0/12 | 0/12 | 3/12 |
| Human splice files flagged | 1/12 | 7/12 | 3/12 | 0/12 | 0/12 | 0/12 | 1/12 |
| Human source groups with a flag | 3/12 | 9/12 | 3/12 | 0/12 | 0/12 | 0/12 | 3/12 |

Reordering the answer letters changed the most likely semantic choice in 38/43 Qwen cases and 43/43 SmolLM2 cases. After testing every order, Qwen abstained on 23 files and selected “Neither” on 12. SmolLM2 abstained on 4 and selected “Neither” on 39. The two-model agreement added zero useful locations in both publishers and both generator families, so this experiment stays outside serving. The fixed Qwen version found three useful locations and flagged eleven human files. Results cover controlled single-sentence replacements in short English government releases.

Each instruction model completed 258 native forwards. The existing classifier completed 36 CPU forwards over 141 unique context inputs. Independent local replay checked 60 publisher ranges, 43 full fixtures, 86 focus ranges, 172 context ranges, all 516 exact prompts and 375 outcome counts. [Aggregate results, sources and commitments](evaluation-panels/writing-option-order-transfer-2026-11-11.json)
