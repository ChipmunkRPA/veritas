# Localizing short rewrites in workplace email

The current three-sentence check and two experimental phrase checks missed all seven known body edits in a fresh workplace-email panel. The current check highlighted one otherwise-human email only after a marked human quote was appended. Both experimental checks avoided that flag and selected zero passages overall. The phrase checks remain in research. [Evaluation receipt](evaluation-panels/writing-email-phrase-persistence-2026-11-11.json)

## Method and sources

The candidate evaluates exact eight-word spans using contextual token responses from the existing Apache-2.0 classifier. Each model input includes the surrounding sentences when their email source role is message body. It stops at a quotation or signature boundary. The unsmoothed comparison uses one center word to score an eight-word span. The persistence comparison takes the smaller of the mean over the central four words and the mean over all eight. This tests whether stronger readings persist across adjacent words, drawing on recent research about local smoothing. The code implements its own fixed-width rule and uses its own data and weights. [Smoothing research](https://arxiv.org/abs/2607.21458) · [Classifier license and checkpoint](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

Each arm could highlight one phrase only if its score exceeded the second-highest source-group maximum among 28 previously used human emails from 14 groups. The current check retained the first six qualifying three-sentence contexts. The two candidate points were frozen before the fresh files were scored. The points are local review settings and carry no authorship probability. Useful localization required at least 0.30 non-whitespace character intersection over union with the known edit.

Eight distinct federal staff members supplied government-duty prose from four Department of Justice releases. The emails date from 2017 to 2019. The retained PDF text-layer lines had no normalized exact 12-token overlap with each other or 45,357 retained prior text entries. Some lines retain OCR irregularities. Each original was tested alone and with a genuine sentence from that source appended as a marked quote. Synthetic multipart EML containers preserve the decoded body and quote positions. The test excludes attachments, alternative HTML and sender authentication. The source roster records exact line locations. [U.S. government works](https://www.copyright.gov/title17/92chap1.html) · [DOJ FOIA library](https://www.justice.gov/oip/available-documents-oip)

Two pinned Apache-2.0 generators each attempted one exact phrase rewrite per source. The initial prompt admitted two of 16 attempts. Before any fresh scoring, a second prompt emphasizing the output format was frozen and tried on all sources. The same completion, length, copy and sentence-geometry checks admitted the first eligible attempt per family and source. This produced seven distinct edits across five source groups, five from Qwen and two from SmolLM. Each was scored in the body, in the body with an added human quote, and in a marked quote beside an unchanged body. [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

## Local results

| Outcome | Current context | Unsmoothed phrase | Phrase persistence |
| --- | ---: | ---: | ---: |
| Human body flagged, plain | 0/8 | 0/8 | 0/8 |
| Human body flagged, with human quote | 1/8 | 0/8 | 0/8 |
| Useful body localization, plain | 0/7 | 0/7 | 0/7 |
| Useful body localization, with human quote | 0/7 | 0/7 | 0/7 |
| Human body flagged when only the quote was edited | 0/7 | 0/7 | 0/7 |

All 37 files were assessed. The persistence score exceeded the paired original’s maximum on all seven rewrites, compared with six for the unsmoothed score. Without applying either cutoff, each candidate ranked an overlapping phrase first on four of seven edits. This retrospective ranking result did not improve first-location accuracy over the unsmoothed candidate and supplies no operating point. The study is small and institutionally related. The serving checks and confidence calculation remain unchanged.

The run made 29 calibration and 40 holdout CPU forwards. A separate model-free readout verified all eight archival positions, 37 exact MIME fixtures, 135 incumbent context inputs, 850 phrase ranges and 85 outcome and transition counts. All 15 original and edited bodies paired with an otherwise identical quoted version had identical candidate measurements.

The [local runner](../backend/scripts/writing_email_phrase_persistence.py) accepts `private-calibration.jsonl` and `private-holdout.jsonl` in a local run directory. Each line binds an EML fixture, exact decoded text, SHA-256, source group and role ranges. Holdout rows add a case kind and half-open body edit offsets. The run directory also contains a `source.json` intake receipt. Run `freeze-calibration`, `calibrate`, `freeze-holdout` and `holdout` in order, passing the run directory and the pinned local classifier snapshot to each stage.
