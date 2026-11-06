# Fresh IML-ViT image-localization comparison

Completed locally on 5 November 2026. **Do not replace the current checks.**
The alternate model missed more edits and produced more false highlights in
both tested formats. No serving setting, customer confidence, or old result was
changed. [Aggregate receipts](iml-vit-localization-evaluation.json) retain the
source, decision, output, checkpoint, and numerical-reference commitments.

## Design

The official checkpoint SHA-256 is
`7631fe852e139d904ad96fd766a37120d703c966cb80dfad70ffae4ff426cce4`.
The maintainer explicitly permits MIT commercial use in
[the checkpoint-license clarification](https://github.com/SunnyHaze/IML-ViT/issues/60#issuecomment-5099083801).
The pinned encoder/FPN source retains its Detectron2 attribution. The local
runtime uses a small independent checkpoint-compatible prediction head and
does not import the upstream decoder file, which cites separately restricted
SegFormer code. All 212 checkpoint keys load strictly, with safe weight loading.
CPU float32 outputs matched the complete publisher reference bit-for-bit on
six consumed development inputs, one original/edit pair per generator.

Development used 24 source-image pairs from the already consumed TGIF v2 panel,
explicitly not fresh validation. Native and JPEG75 gates were fixed separately
before confirmation, each allowing at most one authentic development highlight.
The candidate uses native RGB up to 1024 pixels per side, top-left padding,
ImageNet normalization, and the publisher's model geometry. It crops away only
padding. It never silently resizes or drops an oversized source.

Confirmation used 48 new TGIF/TGIF2 original/edit pairs, 16 each from Adobe
Firefly, SD2, and FLUX Fill. Selection excluded all 144 source IDs from the two
prior local TGIF panels and the earlier web examples. The author test splits
were selected by deterministic hash order before inference. Training-source
overlap remains unknown. Source and edit share exact reference-mask coordinates.
JPEG75 copies preserve those coordinates and count as variants of the same
48 sources, not another independent sample.

The comparison uses the same reliability exclusions, quantized grid, component
filter, and two-region per-map cap. Current PNG means the selected splice check;
current JPEG means the selected fallback localizer. These are component metrics,
not a complete replay of competing UI evidence maps or whole-file AI confidence.

## Held-out results

| Format and check | Genuine images highlighted (48) | Edits with overlapping regions (48) | Edits missed | Mean edited IoU |
| --- | ---: | ---: | ---: | ---: |
| Native PNG — current | 0 | 10 | 38 | 0.14977 |
| Native PNG — candidate | 8 | 8 | 40 | 0.04348 |
| JPEG75 — current | 5 | 9 | 39 | 0.07931 |
| JPEG75 — candidate | 9 | 3 | 45 | 0.01466 |

All 384 confirmation model/format executions completed, with no failed or
out-of-scope inputs. Misses contribute zero overlap. Generator-specific counts,
pixel F1 and source-paired bootstrap intervals are in the JSON. False highlights
are review cues, not false whole-file AI conclusions. Zero highlights on 48
originals is not a guarantee of zero future false alarms.

The current checks remain weak on small local edits. This trial motivates a
separate, frozen multi-scale/tile experiment with the existing commercial
checkpoint. The v4 confirmation cases above have now been consumed and cannot
serve as fresh confirmation for a method designed after seeing these outcomes.

## Reproduction

Scripts: `prepare_tgif_localization.py`, `iml_vit_candidate.py`,
`evaluate_iml_vit.py`, `freeze_iml_vit.py`,
`evaluate_image_localization_baseline.py`, and `summarize_iml_vit.py` under
`backend/scripts/`. Raw files and maps remain in ignored local storage under
`backend/data/localization-eval/iml-vit-candidate/` and
`backend/data/localization-eval/tgif-confirmation-v4/`; they are not redistributed.
The frozen gate record is `frozen-decision.json`. The source plan, complete
receipts and pinned assets are required to reproduce this exact panel; running
a newly selected panel is a separate evaluation. Runtime measurements include
concurrent local work and do not establish a controlled speed comparison.
