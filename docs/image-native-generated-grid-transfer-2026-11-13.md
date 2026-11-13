# Shifted grids and local bit-depth changes on generated image edits

This transfer asks whether three additional native-grid positions improve
localization of generated replacements whose boundaries fall between
the existing 32-pixel cell edges. It also checks whether promoted
8-bit values can explain ordinary interpolation highlights. The two
questions use the same submitted camera pixels, known target masks and
frozen prior texture and interpolation thresholds.
[Prior resampling comparison](image-native-stencil-interpolation-transfer-2026-11-13.md)

Eight checksum-verified CC0 camera RAWs from distinct, previously unused
camera models are developed to linear RGB at 16 bits per channel. The
local Apache-2.0 generator receives a source crop and a fixed request
to insert a yellow star. Each crop is composited at a half-cell-offset
location using a fixed 128-pixel square. Compare sharp and feathered
quadratic 8-to-16-bit transfers, and a separate linear promotion of the
same generated output. Preserve every pixel outside the exact mask.
Ordinary treatments include blur, a local tone adjustment, an even
copy, diagonal cubic interpolation, a rotated bilinear patch and
fractionally shifted 8-bit data promoted by 256 or 257. Untouched,
8-bit and resized full-image controls are counted separately.
[Held CC0 roster](evaluation-panels/image-native-generated-grid-cc0-roster-2026-11-13.json) ·
[Generator terms](image-flux-klein-assets-v271.md)

The new localization arm compares zero and 16-pixel offsets on both
axes. Each view is an exact crop. It projects candidates back to the
native input, retains the highest raw response and applies the prior
interpolation veto without a new cutoff. A useful box needs 0.30
intersection over union and 0.50 changed-pixel recall. Each generated
edge type needs at least two additional locations, at most one additional
ordinary file highlight and no flag on untouched images. All whole-image
8-bit and resized inputs must decline.

A separate bit-depth note measures whether 95% of the codes in every
channel share a remainder modulo 256 or 257 inside the candidate, but
at most 70% do so in the surrounding ring. Both regions must have at
least 24 distinct codes per channel. The frozen rule disallows using
this note to suppress a case if it also occurs on a known linearly
promoted generated replacement. The measurement has zero AI-confidence
weight.

## Held results

| Delivery | Single-grid result | Four-grid result |
| --- | ---: | ---: |
| Generated sharp, useful | 0/8 | 6/8 |
| Generated feathered, useful | 0/8 | 5/8 |
| Generated, linear 16-bit promotion, useful | 2/8 | 7/8 |
| Ordinary highlights | 1/64 | 13/64 |
| Untouched highlights | 0/8 | 0/8 |

The single-grid check highlighted one diagonal cubic edit. The
four-grid check withheld that case but added five rotated local
patches and eight 8-bit-promoted ordinary patches. The 16 separate
whole-image controls all declined. Eight were unsupported 8-bit
rasters and eight resized 16-bit rasters lacked a repeatable
interpolation pattern. The shift improved location recovery for
all three generated deliveries, but exceeded the ordinary-highlight
limit and stays experimental. Every selected generated box matched
the exact 16,384-pixel target and came from the 16-pixel offset on both axes.

All seven selected linearly promoted generated patches met the
code-lattice test. The same clue cannot be used to exclude
generated content. None of the selected ordinary quantized
patches met the additional 24-distinct-values requirement in
every channel. The note is useful only as a narrow observation
about pixel values and has no validated attribution role.

All eight generated outputs showed a visible yellow star inside
the preset target, with one partially clipped by the frozen
composition boundary. Native replay reproduced all 104 cases.
The standard-library reader recomputed overlap and both advancement
decisions using the [public scalar measurements](evaluation-panels/image-native-generated-grid-measurements-2026-11-13.json).
[Audited results](evaluation-panels/image-native-generated-grid-transfer-2026-11-13.json)

## Reproduce

```sh
python3 -B -IS backend/scripts/review_image_native_generated_grid_transfer.py \
  docs/evaluation-panels/image-native-generated-grid-measurements-2026-11-13.json \
  docs/evaluation-panels/image-native-generated-grid-cc0-roster-2026-11-13.json \
  docs/evaluation-panels/image-native-generated-grid-transfer-2026-11-13.json
```

Use the licensed source files from the roster, the prior RawNIND decoder
cache and the verified offline Metal generation runtime. Scoring and
replay are CPU-only. An optional clean Transformers overlay keeps Finder
sidecars out of dependency imports on an external filesystem.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_native_generated_grid_transfer freeze \
  "$run" --cached "$rawnind" --archive "$pixls"
"$cpython3126" -B -m scripts.image_native_generated_grid_transfer generate \
  "$run" --runtime "$mlx_runtime"
for phase in score review; do
  backend/.venv/bin/python -B -m scripts.image_native_generated_grid_transfer \
    "$phase" "$run" --cached "$rawnind" --archive "$pixls" --receipt "$receipt" || exit 1
done
python3 -B -I -S backend/scripts/review_image_native_generated_grid_transfer.py \
  "$run" docs/evaluation-panels/image-native-generated-grid-cc0-roster-2026-11-13.json "$receipt"
```
