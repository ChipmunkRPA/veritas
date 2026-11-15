# Small soft-edged edits in native PNGs

The earlier 16-pixel interpolation cells located sharp 32-pixel edits but
missed the soft-edged examples. A compact fallback compares four 8 × 8
pixel cells at the center of a 32 × 32 crop. The crop includes pixels
around those measurements. This is evidence of a local interpolation
change and carries zero AI-confidence weight.
[Earlier small-region check](image-native-cfa-fine-2026-11-15.md) ·
[Research on camera-pattern localization](https://arxiv.org/abs/2608.20558)

## Measurement

The input remains a 2048 × 1536, 16-bit RGB PNG with the supported
chunk layout and a repeatable native pattern. The normal phase is
estimated from the earlier 32-pixel spatial folds. Existing selected
regions have priority. The new check moves over the original pixels
at four-pixel steps. In each candidate crop, it tests whether the
alternate phase has at least 0.62 interpolation closure and leads
the normal phase by at least 0.25 in each of the four inner cells.
The basic rule needs at least three supporting cells.

The second rule needs at least two supporting inner cells and normal
closure in six of eight exterior cells, including at least one on
each side. Each rule's two points come from the largest and second
largest source-group maxima among ordinary cases on six development
cameras. Both points were zero because no ordinary development view
qualified. The points and source roles were fixed before the held
cameras were measured. Each rule takes the strongest eligible crop.

## Held results

Two groups of ten camera models supplied retained six-pixel blends
and new off-grid cases at two blend widths. The offset crops start
between the search grid lines. Known masks use the exact changed
pixels. A useful crop covers at least half the changed pixels and
reaches 0.30 intersection over union. These are staged odd-pixel
copies from the same image. Genuine source RAWs come from the
prior CC0 rosters.

| Staged 32-pixel edit | Earlier useful | New inner-cell check | With exterior recovery |
| --- | ---: | ---: | ---: |
| Aligned six-pixel blends | 0/20 | 16/20 | 9/20 |
| Offset six-pixel blends | 0/20 | 12/20 | 9/20 |
| Offset nine-pixel blends | 0/20 | 12/20 | 10/20 |

The inner-cell check also added one useful sharp crop among 20 known
sharp edits. Earlier selected regions stayed in place, including the
48- and 64-pixel feathered results. Across
160 assessed ordinary deliveries, neither rule added a crop. Those
deliveries include native images, tone and blur changes, translations,
and new soft-edged copies preserving the original interpolation phase.
The 60 resized, adaptive-development and JPEG-derived controls stayed
unassessed. Each camera supplied one source image, and the different
deliveries reuse those images.

The inner-cell check locates 40 of 60 staged small feathered edits.
The surrounding-cell requirement loses twelve of those locations and
the two rules have the same observed ordinary-control outcome. This
supports a bounded local trial of the first rule.

## Local trial

The development setting `VERITAS_IMAGE_NATIVE_CORE_REVIEW=true`
enables the compact check alongside the native and fine interpolation
checks. It runs after the earlier checks complete without a selected
region. Existing compatible exports still replay with their original
schema.

Four simulated USD 9.99 checkouts tested one soft-edged edit and a
matched ordinary copy with the option off and on. No analysis job
entered the queue before payment. The new check marked columns
**[1388, 1420)** and rows **[448, 480)** on the staged edit, with all
four inner cells supporting the alternate pattern. The PDF contains
the same outline and crop as the evidence API and JSON. The crop
matches the submitted 16-bit pixels exactly. Signed source verification
replayed the measurement and the images. The same-phase copy showed
no region, and the automated assessments agreed with the setting off
and on.
[Local application receipt](evaluation-panels/image-native-cfa-compact-core-app-2026-11-15.json)

## Replay

The receipt records exact crop coordinates, the supporting cell
measurements, source hashes, and results by camera role and edit type.
The portable reviewer recalculates the thresholds, verifies the native
placement and compares the candidate to the earlier selections.
[Comparison receipt](evaluation-panels/image-native-cfa-compact-core-2026-11-15.json.gz)

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.image_native_cfa_core_transfer review \
  docs/evaluation-panels/image-native-cfa-compact-core-2026-11-15.json.gz
```

For native replay, use the pinned decoder wheel and original RAWs
described in the earlier check, with new output paths.
[Decoder and source setup](image-native-cfa-fine-2026-11-15.md#reproduce)

```sh
backend/.venv/bin/python -B -m scripts.image_native_cfa_core_transfer run \
  "$run/receipt.json.gz" "$cache" "$archive"
backend/.venv/bin/python -B -m scripts.image_native_cfa_core_app \
  "$app_run" "$run/receipt.json.gz" "$cache" "$archive"
```
