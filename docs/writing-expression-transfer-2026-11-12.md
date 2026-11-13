# Writing-style corroboration for one-sentence rewrites

The combined writing-style and position cue recovered zero of 24 known rewrites on a fresh Open Research Europe panel. It added two false genuine work groups. The existing check located four rewrites and flagged five of 12 genuine work groups. This cue stays outside the application. [Fresh evaluation receipt](evaluation-panels/writing-expression-ore-transfer-2026-11-12.json)

| Fresh outcome | Existing context | Expression alone | Expression and position | Existing plus expression and position |
| --- | ---: | ---: | ---: | ---: |
| Qwen locations | 3/12 | 0/12 | 0/12 | 3/12 |
| Granite locations | 1/12 | 0/12 | 0/12 | 1/12 |
| Genuine originals flagged | 4/12 | 1/12 | 2/12 | 6/12 |
| Genuine later versions flagged | 4/12 | 1/12 | 1/12 | 5/12 |
| Genuine works flagged in either version | 5/12 | 1/12 | 2/12 | 7/12 |

All 24 edits had eligible exact positions. The combined cue proposed four other sentences in mixed files. Those positions were also present in genuine controls from the same two works. The previous position check found two of 24 edits at its short-passage point while flagging five genuine works. At its article-calibrated point it found zero and flagged one genuine work.

## Controls and frozen method

The Open Research Europe intake used complete dated Crossref metadata and 68 version-specific publisher XML files. Twelve works qualified with article-level CC BY 4.0 terms, equal recorded bylines, pre-November 2022 dated releases, nonoverlapping exact text fragments, and a genuine editorial change in the selected body excerpt. Each work supplied an original, the real later version, and one controlled exact sentence replacement from each of two locally installed Apache 2.0 generators. The journal was held out from development and shares a publishing platform with several development journals. [Crossref metadata API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) · [Prior source method](writing-long-position-transfer-2026-11-12.md)

The expression cue measures function-word categories, grammatical bigrams, punctuation, and runs of content-word placeholders. It compares the focal sentence with both immediate neighbors and the remaining native sentences. The feature vector omits content-word identities. A balanced, regularized linear fit uses 24 known edited centers and 164 genuine centers on previously exposed work groups. A separate 14-work genuine calibration sets each point from the second-highest maximum across a work's paired versions. One branch adds the previously frozen position contrast. Research on mixed-origin evidence motivates separate content and expression signals, and findings on AI editing motivate testing transfer with actual revisions. [Mixed-origin evidence research](https://arxiv.org/html/2608.27380) · [AI editing research](https://aclanthology.org/2026.emnlp-main.1725/)

The initial PLOS diagnostic used a malformed word regular expression. I excluded its measurements, restored the intended Unicode tokenizer and refitted only on the unchanged development partition. The corrected fit and points were frozen before any Open Research Europe detector scoring. On the already exposed PLOS diagnostic panel, the corrected combined cue retained all four useful locations found by the old position point, with zero of that rule's four genuine-work flags or seven wrong mixed-file locations. Its controls included a constructed substitution using real prose from elsewhere in the same article. That PLOS outcome is diagnostic, and the fresh real-revision results above determine the product decision. [Corrected diagnostic receipt](evaluation-panels/writing-expression-plos-transfer-2026-11-12.json)

## Replay

The independent arithmetic reviewer verifies the fitted objective, the points, exact known positions and the position-only ablations. Native replay verified 36 publisher XML files, 132 within-panel work-pair checks, and 4,360 saved input occurrences. The receipts carry hashes, attribution, spans and derived measurements. Source prose and generated rewrites stay local.

```sh
python3 -B -I -S backend/scripts/review_writing_expression_transfer.py \
  docs/evaluation-panels/writing-expression-ore-transfer-2026-11-12.json
```

The model-free feature extractor is in `backend/scripts/writing_article_expression_features.py`. Call `feature_rows(text)` with a normalized plain-text excerpt. It returns the source-linked sentence units and expression measurements. Running the pretrained position contrast requires the existing pinned writing encoder.
