# Two writing detectors on the same source ranges

The trial asks whether a second classifier can support an existing highlight or recover a missed edit in a single submitted version. Both checkpoints see the same exact text. Their weights and training pipelines differ, while the inputs and many linguistic cues are shared. Recent research shows why local context matters and why ordinary editing controls are essential. This experiment uses its own selection rules and imports none of the papers' code, data or weights. [Sentence and context research](https://aclanthology.org/2025.emnlp-main.268/) · [Professional editing and false positives](https://aclanthology.org/2026.emnlp-industry.109/)

## Frozen inputs and comparisons

The panel combines 57 four-paragraph Census and Copyright Office files and 43 four-sentence FAA and USDA Farm Service Agency files from retained source-bound experiments. There are 28 publisher source groups and 100 plain-text files. Each group includes the original, a length-matched human splice, and one or more known replacements by a locally run generator. The scored panel contains 56 controls and 44 admitted replacements across OLMo, Granite, Qwen and SmolLM. The original reports record the generator admissions, exact source anchors and licensing. These sources had prior project exposure, and training exposure is unknown. [Paragraph source panel](writing-grammar-seams-2026-11-11.md) · [Sentence source panel](writing-option-order-transfer-2026-11-11.md)

The runner reconstructed every serving three-sentence context, including the two-sentence edges. It checked exact spans, input hashes and complete token coverage against the saved incumbent observations. The second checkpoint read those same inputs on CPU without network access. Both checkpoint cards declare Apache-2.0, and the runner checks the second checkpoint's weight, license and configuration commitments. The project uses the pinned weights only. It did not retrieve or use the second checkpoint's training corpus, whose publisher lists mixed terms for its source material. [Current checkpoint](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small) · [Second checkpoint](https://huggingface.co/rasbt/ai-text-detector-modernbert) · [Training corpus terms](https://huggingface.co/datasets/rasbt/human-vs-ai-50k)

The current and second-only comparisons retain their first six positive windows in source order. The agreement comparison keeps a current window only when the second checkpoint also responds positively. It does not backfill dropped windows. The rescue comparison keeps every current window and may add the second checkpoint's strongest complete three-sentence window. The focus needs at least 12 words. For longer files, only the maximum window in each focus paragraph competes. In a single paragraph, the interior focuses compete directly. A new location requires a positive second response with at least twice the odds of the next highest candidate. All selected regions cover the exact scored inputs. These are internal comparisons of raw model outputs and supply no public confidence threshold.

A useful location must overlap the exact replacement at an intersection over union of at least 0.30 in non-whitespace characters. All flags on human controls are counted, with both file and source-group counts. The frozen agreement criterion requires no lost useful location and at least two cleared human groups. The frozen rescue criterion requires a gain in at least two source groups, two publishers and two generator families, with at most one additional flagged human group.

## Results

| Check | Useful edit locations /44 | Human files flagged /56 | Human source groups flagged /28 |
| --- | ---: | ---: | ---: |
| Current | 14 | 17 | 11 |
| Second alone | 35 | 49 | 26 |
| Require same-window agreement | 14 | 14 | 9 |
| Current plus second-check rescue | 28 | 46 | 26 |

Agreement kept all 14 useful current locations. The three removed human flags and both cleared source groups came from the Copyright Office. The rescue gained 14 useful locations across ten groups and all four generator families. It also added flags in 15 human source groups. The second check by itself gained 23 useful locations and lost two, with a higher false-alarm count. Agreement met the frozen local pilot criterion. The other options remained experimental. The small improvement is concentrated in one publisher, and files within each group are related. This comparison does not estimate a customer false-alarm rate.

The second classifier read 453 unique inputs in 114 CPU forwards and the scoring stage took 341 seconds. A separate model-free replay reassembled 683 exact input occurrences and reproduced 374 counts. [Aggregate results and commitments](evaluation-panels/writing-checkpoint-agreement-2026-11-11.json)

## Local app

In a development instance using candidate text mode and three-sentence windows, enable `VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW=true` for Full and Deep profiles. The option covers mapped English excerpts of 68–420 words. It revisits up to the first six AI-like contexts from the pinned first check. Supported plain text, top-level Word body paragraphs, PDFs with mapped selectable text on every page, and unquoted plain email use four paragraphs or one paragraph of four sentences. When a trailing quotation or signature follows a supported plain-text reply, the check uses only contexts wholly inside its contiguous one- to six-paragraph body. The single-paragraph body still requires four sentences. A missing or incomplete second response stays unavailable. The second check carries zero authorship-confidence weight. [Format evaluation](writing-agreement-document-format-transfer-2026-11-12.md) · [Reply-body evaluation](writing-email-body-scope-transfer-2026-11-12.md) · [Additional reply layouts](writing-reply-native-batches-2026-11-12.md)

A local Full-profile replay used the same selected text fixture through checkout, the paid job, the evidence API, the JSON export and a rendered PDF. The simulated Stripe event released the job without a real charge. Both native checks ran in the first replay. The second check took about 2.5 seconds and recorded two agreement ranges and one disagreement. A second replay used an isolated local model cache with the second checkpoint absent. All three ranges remained available as primary contexts and were marked as lacking a second response. The overall result stayed inconclusive in both replays.

## Local replay

The retained panel directories and pinned second checkpoint are required for native scoring. The separate reviewer uses only the standard library and the saved response file. It reassembles each model input from original source ranges, checks coverage and counts, and verifies the bitset overlap independently. No customer text, raw article text or weights appear in the public aggregate.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for stage in freeze score readout
do
  backend/.venv/bin/python -B -m scripts.writing_checkpoint_agreement "$stage" "$run" "$panels" "$checkpoint" || exit
done
backend/.venv/bin/python -B -m scripts.review_writing_checkpoint_agreement "$run" "$panels"
```
