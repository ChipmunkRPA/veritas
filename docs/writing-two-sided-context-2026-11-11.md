# Two-sided sentence context on technical prose

The earlier short-message transfer recovered more useful locations with a fitted three-sentence score and added three human flags. This comparison tests whether the response follows a sentence when either neighbor is removed. Recent segmentation research separates local evidence from changes in a document's score sequence. This experiment implements a simpler matched comparison with the current local classifier. It uses the existing pinned Apache-2.0 weights and no code or data from that research. [Prior NASA transfer](writing-focus-publisher-transfer-2026-11-11.md) · [Local-evidence change-point study](https://arxiv.org/abs/2609.33787) · [Pinned checkpoint](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

## Fixed comparison

For each interior sentence of at least 12 words with two neighbors in the same paragraph, score the full context, the sentence with its previous neighbor, the sentence with its next neighbor, and each neighbor alone. The new fit uses the full-context log odds, the lower of the two paired-context log odds, and the smaller change relative to each corresponding neighbor alone. Both fits use the same eligible sentences, balanced source weights and regularization. The full-window fit uses its first feature alone. Separate consumed groups fix each review point at the largest assessed human source maximum. The confirmation uses a single point per fit. The current raw check and a version that displays its exact selected sentences both serve as references.

Fit used 31 eligible sentences across six previously used work-email source groups, including nine known rewrites. Calibration considered 13 other earlier work-email groups from three packets. Seven had an eligible interior sentence and six were unassessed. The fitted coefficients and thresholds were recorded before the fresh classifier pass. These review points have no population false-alarm guarantee.

## Fresh sources

Six distinct NIST authors provided dated technical prose from 2018–2021 about dimensional measurement, instrument standards, spectral calibration, electrical standards, historical measurement methods and statistical inference. Each file has four consecutive sentences from one body paragraph. Each source also provides a length-matched donor sentence elsewhere in the same article for a controlled ordinary replacement. Article captions, quoted third-party prose, footers and comments are excluded. The originals and donor controls had no normalized exact 12-token overlap with another group or 45,307 retained project entries. The authors share a publisher, and model pretraining exposure is unknown. [Publisher rights](https://www.nist.gov/copyrights-disclaimers) · [Exact source locations and outcomes](evaluation-panels/writing-two-sided-context-2026-11-11.json)

Two pinned Apache-2.0 generators attempted six rewrites each offline. Three Qwen outputs passed the fixed complete-sentence, length, copy and overlap rules. None of the initial SmolLM outputs passed. A new SmolLM prompt and seed were frozen before scoring any of the fresh cases. Three of its six outputs passed the same rules. All admitted edits preserve the exact original text outside one sentence. The six rewrites span six distinct source groups. [Qwen terms](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM terms](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

## Local results

| Outcome | Current context | Current focus | Fitted full window | Two-sided |
| --- | ---: | ---: | ---: | ---: |
| Human originals flagged | 4/6 | 4/6 | 2/6 | 3/6 |
| Genuine sentence replacements flagged | 2/6 | 2/6 | 0/6 | 2/6 |
| All human files flagged | 6/12 | 6/12 | 2/12 | 5/12 |
| Human source groups with a flag | 4/6 | 4/6 | 2/6 | 4/6 |
| Qwen rewrites usefully localized | 0/3 | 0/3 | 0/3 | 1/3 |
| SmolLM rewrites usefully localized | 1/3 | 1/3 | 2/3 | 2/3 |
| All rewrites usefully localized | 1/6 | 1/6 | 2/6 | 3/6 |
| AI files flagged only at a wrong location | 0/6 | 1/6 | 0/6 | 0/6 |

Useful localization requires at least 0.30 non-whitespace character intersection over union with the known rewrite. The two-sided fit kept the one useful current location and added one from each generator family. It removed one flag on an original and added no new human file flags against the current check. The matched full-window fit found two locations and flagged three fewer human files than the two-sided fit. None of the flags on human replacement files overlapped the inserted donor sentence.

All 18 admitted TXT files passed the app's extraction and exact text checks. The classifier made 39 native forwards, including 15 for fresh cases. A separate arithmetic readout checked the original and altered source boundaries, 216 context geometries, all six rewrite foci, 124 outcome counts and four human group counts. No raw source or generated text is included in the public aggregate.

The two-sided method met the predeclared criteria for a larger panel. The stronger false-alarm result of the matched full-window fit leaves an open tradeoff. This is one small technical publisher, and the result supports a new source-disjoint comparison. Neither fitted method is enabled in serving.

The [local runner](../backend/scripts/writing_two_sided_context.py) accepts three private newline-delimited JSON files named `private-fit.jsonl`, `private-calibration.jsonl`, and `private-holdout.jsonl`, plus `source.json` with a source roster and exposure inventory. Constructed rows contain case and group IDs, the role, the exact decoded text and SHA-256, `ai_spans`, `ordinary_spans`, and optionally a local source fixture path. Codepoint intervals are half-open. Run `python -m scripts.writing_two_sided_context freeze RUN_DIR` from `backend`, followed by `run RUN_DIR --snapshot LOCAL_MODEL`. Reviewers can build separate lawful sources with the same format.
