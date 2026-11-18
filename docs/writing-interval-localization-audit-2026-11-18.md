# Why the writing check misses some edited passages

This audit replays the saved writing responses from three earlier comparisons. It checks whether the sentence boundaries can represent each known edit, whether any qualifying range reaches the existing operating point, and whether that range survives selection. It also tests a way to split a broad highlight. Every comparison uses the same recorded responses and source coordinates. The audit makes no change to the app or to overall confidence.

The two main panels contain 49 retained and 40 previously held matched pairs. They include historical prose and Federal Reserve reports. A third panel contains 18 matched pairs and 56 additional ordinary documents, including diplomatic correspondence. The source studies document the rights review, source groups, and limits of the generator labels. The panels were already used in earlier comparisons, so this is a retrospective replay. [Within-document calibration and matched panels](writing-local-baseline-transfer-2026-11-16.md) · [Document-profile comparison and controls](writing-profile-calibration-2026-11-16.md)

## Where the check loses edits

A highlight must cover half the labeled non-whitespace characters and achieve at least 0.30 intersection over union. Each highlight can match one label. All 55 labels in the fresh published-layout panel can be expressed as native sentence ranges under that rule. The pooled and within-document checks located 22 before the optional economic report confirmation.

| Cause of a miss | Fresh published layout |
| --- | ---: |
| A supported candidate existed below both frozen operating points | 24 |
| An above-point candidate lost within its own selector | 5 |
| An above-point local selection was suppressed when retaining existing ranges | 1 |
| A selected range also matched another label and received credit for that label | 3 |

Historical prose accounted for 22 of the 24 below-point misses. Its 20 untouched files received zero highlights. The pooled and local checks located 16 of 23 economic report edits and flagged 10 of 20 unchanged reports. Optional economic report confirmation located 13 of those edits and reduced the flagged reports to three. The localization and ordinary-alert tradeoff differs sharply by source type.

## Splitting an existing highlight

The split trial replaces at most one displayed range per file with two shorter ranges inside it. Both must independently pass their existing operating point and any applicable report confirmation. The trial keeps at least one unselected sentence between the two ranges and observes the existing limit of six. If several splits qualify, it prefers the pair with the largest combined excess over the existing operating point. Comparisons below use the published layout and start from the checks with optional report confirmation enabled.

| Comparison | Labeled locations before → after | Ordinary files flagged before → after | Unmatched ranges in edited files before → after |
| --- | ---: | ---: | ---: |
| Retained | 35 → 39 of 98 | 2 → 2 of 49 | 5 → 14 |
| Fresh | 19 → 20 of 55 | 3 → 3 of 40 | 5 → 9 |
| Document-profile comparison | 13 → 13 of 22 | 10 → 10 of 74 | 2 → 5 |

The retained split gained ten locations and lost six. The fresh split gained one, lost zero, and added one range to an already-flagged ordinary file. The separate comparison added zero ranges to its 40 diplomatic controls. The receipt also records the whitespace-neutral layout, which produced the same fresh economic results and zero fresh historical gains. The measured benefit does not support changing the selector. Improving how the check scores genuine and edited historical prose is a more promising target than changing how it draws the ranges.

## Replay

The audit receipt contains aggregate counts and hashes of the three existing numeric input receipts. The reviewer needs only those receipts and the repository's version of the passage arithmetic. It uses no model weights, private text, or network access. [Audit receipt](evaluation-panels/writing-interval-localization-audit-2026-11-18.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_interval_localization \
  docs/evaluation-panels/writing-interval-localization-audit-2026-11-18.json
```
