# Exact JPEG delivery identity for an earlier image copy

This comparison covers a narrow same-framing case that the existing integer alignment can decline. An earlier RGB PNG is encoded under the submitted JPEG's fixed delivery settings. The comparison checks the two decoded RGB grids at the same positions. A broad exact match with unique detailed anchors supports the zero-offset alignment and allows the existing difference-mask routine to run. The route has no AI-authorship score. [Earlier-copy workflow](image-reference-comparison-2026-11-11.md) · [Prior measurement of unresolved pairs](image-patch-proposal-audit-2026-11-12.md)

## Profile and admission

Both inputs must be upright and have identical dimensions from 256 to 768 pixels per side. The earlier file must be a single-frame 8-bit RGB PNG without a color profile, transparency or color-conversion declarations. The submitted file must be a baseline RGB JPEG with 4:2:0 sampling and the exact standard quality-75 tables. The method leaves other deliveries to the existing checks.

Partition the decoded grids into complete, nonoverlapping 16-pixel blocks at the source origin. At least 65% must agree in every RGB byte. At least 24 matching blocks must be unique in each complete-block grid and have a channel sample range of at least 24. The anchors must include at least two blocks per zone in six or more of the nine fixed zones, with every zone row and column represented. The right and bottom partial blocks do not supply anchors. The exact anchor hashes and decoded reference-grid hash are retained in the private replay. A second encoder comparison verifies pixel equivalence with the production reference codec.

The evaluator considers the new result only after the existing comparison has declined. Existing comparable output takes precedence. The candidate uses the existing difference-mask routine, exact submitted-grid placement and identical crop coordinates on both copies. A useful region needs 30% intersection over union and 50% coverage of the known pre-JPEG edited pixels.

## Panels and controls

Twenty development source groups and twelve distinct confirmation groups come from the earlier licensed patch studies. The panels use different source photographs and credited photographers from one another. The project previously evaluated both panels under other methods. The admission rule and continuation requirements were fixed before scoring the development or confirmation panel.

Each source supplies the same eight matched JPEG75 views as the earlier study. Add six controls per photo, including a resized unrelated image with the exact same dimensions, whole-frame displacements of two and sixteen pixels, a horizontal flip, and two ordinary JPEG-to-JPEG recompressions. Separate synthetic controls use exact period-16 repeated tiles, the same image shifted by a full period, a locally changed periodic image and an indistinguishable mirrored image. These probe whether many exact tiles with little spatial specificity can establish alignment. [Development source roles and rights](image-three-class-patch-probe-2026-11-12.md) · [Confirmation source roles and rights](image-three-class-patch-fresh-2026-11-12.md)

The frozen continuation rule needs two additional feathered regions in confirmation and zero admitted wrong-photo or transformed-image controls, added unchanged or recompression highlights, newly misplaced edit masks, or lost incumbent results.

## Local evaluation

| Panel | Useful sharp, current → combined | Useful feathered, current → combined | Newly compared unchanged | Wrong-photo or transformed pairs admitted | Recompression pairs admitted |
|---|---:|---:|---:|---:|---:|
| Development | 14/20 → 20/20 | 14/20 → 20/20 | 5 | 0/80 | 0/40 |
| Confirmation | 8/12 → 11/12 | 8/12 → 11/12 | 3 | 0/48 | 0/24 |

The three additional confirmation groups include one Kandinsky patch and two Stable Diffusion 1.5 insertions. All eight newly compared unchanged files across the two panels had empty masks. One confirmation photo remained outside the candidate scope. There were zero newly misplaced edited masks and no change to any previously comparable result. Confirmation added eight real conventional difference highlights on tone, copy, blur or local autoencoder renditions. All four synthetic repeated or symmetric texture cases were declined. The ordinary single-pass JPEG tests independently reproduced every decoded pixel on all 32 original photos.

The development and confirmation runs processed 280 and 168 JPEG deliveries in 24.61 and 15.38 seconds. Separate repeat executions reproduced all 448 private result records and all four synthetic decisions. The serving module reproduced those 448 records after compacting the anchors into a count and SHA-256 commitment. No models ran. [Frozen result and replay ledger](evaluation-panels/image-reference-exact-jpeg-2026-11-12.json)

## Local app and exports

On a development server with `VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED=true`, set `VERITAS_IMAGE_REFERENCE_EXACT_JPEG_IDENTITY=true` and restart the API. Open a completed image analysis, select **Compare with an earlier copy**, then provide the separate earlier PNG. A comparable result with no retained regions means the alignment succeeded and the difference rule selected no region. An unassessed result means the inputs or their alignment could not support a comparison. The copies and processing may still contain differences that the selection rules miss.

The supplemental JSON keeps the full native framing and compact exact-block commitments. The PDF records the exact block counts and includes each native crop, difference mask and highlighted exhibit when present. The offline reviewer recomputes the delivery and anchors from both original files, checks the full comparison metadata, then verifies the retained mask and native crop pixels.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_image_reference_oriented_resize_export \
  "$export" "$earlier_png" "$submitted_jpeg"
```
