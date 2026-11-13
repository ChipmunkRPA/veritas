# Comparing the changed sentence with its full context across drafts

This trial tests whether a sentence-specific response can add useful locations to the existing writing review when the user has a verified earlier draft. Each comparison measures the same known change using the revised sentence alone and its complete native context. The unchanged sentences on either side are identical in both versions. The app's current paired comparison measures the full context. [Current paired comparison](writing-paired-revision-replication-2026-11-11.md)

## Sources and method

The internal operating points use nine dated F1000Research calibration groups. The two held sets use 21 other Wellcome Open Research and Gates Open Research works with 32 admitted Qwen or SmolLM2 edits, plus ten additional F1000Research works with 14 admitted Granite or OLMo edits. The published originals and ordinary revisions provide two genuine controls per work. Both versions carry article-level CC BY 4.0. Each artificial case changes one known sentence and preserves the rest of its excerpt. The works and exact classifier input hashes are disjoint from calibration. The source material had prior project exposure. Wellcome, Gates and F1000Research share a publisher platform. [Calibration sources and terms](writing-reference-revision-transfer-2026-11-11.md) · [Wellcome and Gates sources and terms](writing-journal-body-transfer-2026-11-11.md) · [Separate F1000Research sources and terms](writing-source-aligned-transfer-2026-11-12.md)

Both pinned Apache-2.0 checkpoints read every exact center and exact context. Center-only uses the smaller before-to-after log-odds change across the two checkpoints. The joint candidate uses the smallest change across both checkpoints and both input scales. Each candidate requires strict excess of its own second-highest genuine calibration group. The prior paired direction uses a doubling of odds in both context readings. The incumbent comparison uses existing passage locations and the existing paired rule where the actual earlier-draft route accepts the exact layout. A useful location covers at least half of the known edited non-whitespace characters and reaches 0.30 intersection over union. The follow-up criterion requires two useful additions from distinct held works across both generator families, at most one extra genuine group and corroboration on the other held panel. This comparison borrows source alignment and local evaluation concepts from recent work. It uses its own code, licensed sources and pinned models. [Checkpoint terms](writing-checkpoint-agreement-2026-11-11.md#frozen-inputs-and-comparisons) · [MixDetect](https://arxiv.org/html/2609.32625v1) · [Operation-guided editing benchmark](https://arxiv.org/html/2606.06481v1)

## Held results

| Review | Additional useful Wellcome/Gates locations / 32 | Additional genuine work groups / 21 | Additional useful F1000 locations / 14 | Additional genuine work groups / 10 |
| --- | ---: | ---: | ---: | ---: |
| Existing paired context rule with a wider layout | 1 | 0 | 1 | 0 |
| Context only, calibrated on the same genuine works | 7 | 1 | 4 | 1 |
| Exact center only | 12 | 5 | 8 | 3 |
| Exact center and full context | 6 | 0 | 4 | 0 |

The existing app and passage checks found six and four useful locations respectively on the two panels. The joint comparison at the broader research scope added locations from all four generator families and preserved prior locations by union. The additional calibrated context-only comparator was calculated after the initial run using the same fixed selection rule. It flagged two more genuine groups and found one more useful location. Under the app's existing single-paragraph eligibility, the calibrated context and joint methods added the same six exact cases and neither added a genuine flag. The app keeps the existing single-paragraph rule.

## One preserved paragraph break

The optional local draft comparison accepts an otherwise eligible single-sentence edit when exactly one existing paragraph break separates the target and one of its immediate neighbors. Both drafts must retain the same sentence-to-paragraph layout. The unchanged neighbors and all linked wording changes must bind to the exact original Unicode ranges. The route still requires two short English TXT drafts. The paragraph-break check uses both scales and publishes their uncalibrated before-and-after responses alongside the exact ranges. It carries zero authorship confidence weight.

The additional scope admitted 12 held generated and eight ordinary revised pairs. Exact ordered app batches reproduced every decision from the pooled readings. The joint check found four of 12 edits across three works and three generator families, with zero of eight ordinary revisions flagged. Calibrated context alone found five edits and flagged one ordinary revision. The controlled trade-off is one missed edit and one avoided ordinary flag. All known locations are exact source spans. This test measures a narrow draft comparison and does not estimate single-file detector accuracy.

Set `VERITAS_DOCUMENT_PAIRED_BOUNDARY_REVIEW=true` in a local instance already configured for the experimental earlier-draft check. The owner must choose the existing paired-writing option after uploading an earlier draft.

## Replay

The portable receipt contains 126 case views, 252 complete input hashes, 504 raw responses and 22 exact app-ordered batches. Both models made 63 CPU forwards for the pooled study. The app-ordered replay ran 22 forwards per model. Original excerpts, generated text, and native model inputs stay local. [Study and exact native batch receipt](evaluation-panels/writing-paired-center-transfer-2026-11-12.json)

```sh
python3 -B -I -S backend/scripts/review_writing_paired_center_transfer.py \
  docs/evaluation-panels/writing-paired-center-transfer-2026-11-12.json
```

A separate implementation in Node.js reproduced the 126 case views, all 504 pooled readings, the two held result tables, and all 22 native batch decisions. It also matched the four public app case records to their study cases. With the retained originals and actual JSON exports, it verified eight original draft files, their exact Unicode ranges, and their saved check directions. The replay rejected a changed batch decision, a reversed batch, shifted public and app ranges, and changed source bytes. Node.js recomputes the saved evidence without running a model or providing an outside evaluation. [Independent runtime receipt](evaluation-panels/writing-paired-center-transfer-node-2026-11-12.json)

```sh
node backend/scripts/review_writing_paired_center_transfer_node.mjs \
  docs/evaluation-panels/writing-paired-center-transfer-2026-11-12.json \
  docs/evaluation-panels/writing-paired-center-transfer-app-2026-11-12.json
```

An optional third argument is the local paid-replay directory. Each case subdirectory must contain `comparison.json`, the historical `old.json` export, `earlier.txt`, and `analyzed.txt`. The reviewer reads these files locally and emits counts and hashes.

## Local app verification

Four ordinary Full-profile analyses ran after four simulated payments. No unpaid event queued analysis. The draft route recorded joint increases for three known generated edits and no joint increase for a genuine F1000Research revision with the same paragraph geometry. Opting out restored the earlier behavior, and the comparison left each completed analysis digest intact. The downloaded comparison PDF linked the same changed text as the JSON. Both exact ranges and both response scales replayed from the original draft pair. An old schema-2 out-of-scope result replayed against the same files. The comparison route exports standalone PDF and JSON. The offline source replay checks both supplied original files and the saved response arithmetic. Chrome uploaded an earlier draft, followed the wording link and downloaded PDF and JSON equal to its API responses. [Local app receipt](evaluation-panels/writing-paired-center-transfer-app-2026-11-12.json)
