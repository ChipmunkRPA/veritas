# Show the most supported writing passage first

An opt-in local review puts one existing writing passage first when the same check gives an AI-like response to both a complete three-sentence input and a separate read of its whole paragraph. The most AI-like qualifying sentence-context response comes first, with source order breaking ties. All passages keep their original IDs and source coordinates. The other passages stay in their original order, and all supporting and opposing responses remain visible. The two readings use the same classifier and add zero weight to authorship confidence.

The pilot covers mapped English `.txt` sources with four paragraphs and 145–470 words, using the pinned candidate with a three-sentence window. To try it in local development, enable both `VERITAS_TEXT_ISOLATED_PARAGRAPH_REVIEW=true` and `VERITAS_TEXT_PARAGRAPH_EVIDENCE_PRIORITY=true` in candidate mode. The second setting defaults to false. Analysis view, PDF and JSON use the same saved passage order and refer to the same evidence ID. Original-source verification rechecks offsets and recorded inputs without a model run.

## Retrospective replay

The replay used the actual ordering function on 78 previously scored plain-text fixtures, with existing saved responses and a single-check baseline capped at the first six qualifying cards in source order. A useful first card covers the known generated rewrite at an intersection over union of at least 0.30 in non-whitespace characters. [Audited aggregate](evaluation-panels/writing-paragraph-priority-2026-11-11.json)

| Prior panel | Useful first card before | Useful first card with priority | Human files with retained passages in both orders | Human files with a different first card |
| --- | ---: | ---: | ---: | ---: |
| Oral histories and public addresses | 3/16 | 11/16 | 24/27 | 8/27 |
| Court opinions and prepared speeches | 3/11 | 10/11 | 15/24 | 7/24 |

The 15 added useful first locations came from nine source groups. No previously useful first location was lost. The replay changed which card readers see first in 15 human files, while all their existing flags stayed visible. Seven unavailable rewrites had already failed the original intake. These panels share their classifier and use related files within each source group. The results measure navigation for these fixtures, and the single-check replay does not estimate a multi-check deployment's localization rate. [Speech transfer](writing-speech-paragraph-transfer-2026-11-11.md) · [Court and prepared speech](writing-paragraph-replication-2026-11-11.md)

A separate local smoke test ran the pinned classifier on one selected oral-history rewrite from the replay. The saved lead moved from `T01` to `T04`, which overlapped the known edit at the required extent. All five saved cards and the decision were identical across both orders. The saved opt-in packet passed original-source replay through the app evidence loader.

To reproduce an aggregate from a retained local run directory, pass its original public receipt to the model-free replay.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.replay_document_paragraph_priority \
  "$run_dir" docs/evaluation-panels/writing-speech-paragraph-transfer-2026-11-11.json
```
