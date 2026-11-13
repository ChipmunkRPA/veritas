# Locate small local rotations in a native PNG

**Local pixel interpolation** can show a region with an interpolation
pattern consistent with a small local rotation. The same section shows
the region on the submitted image, its original 16-bit crop, and the
exact source coordinates. The red and blue channels must support an
angled pattern in both parts of the crop. This is a local resampling
observation and contributes zero AI confidence.

This development option applies to a bare 2048 × 1536 RGB PNG with
16-bit channels and a repeatable interpolation pattern. The established
phase-shift observation keeps priority when it finds a region. The
rotation measurement tests four grid origins on the original pixels and
requires distinct off-axis support in the selected crop. An image may
contain an edit that produces no qualifying measurement. The output
does not estimate a rotation angle.
[Previous spectral rotation test](image-native-rotated-stencil-transfer-2026-11-13.md)
· [Resampling research](https://doi.org/10.1109/TPAMI.2026.3721972)

## Held comparison

Thirteen CC0 camera models from the prior interpolation transfer supplied
native pixels. These sources were disjoint from both earlier rotation
cohorts. They had prior project exposure. The newly authored local
rotations use known masks and range from 2.1° to 5.56° in absolute
angle. The small angles, 128-pixel patches and controlled rendering
bound the result. The 26 generated controls reuse the prior exact
source-conditioned replacement bytes.
[Source roster](evaluation-panels/image-native-stencil-interpolation-cc0-roster-2026-11-13.json)

| Held case | Useful locations or flags |
| --- | ---: |
| Bilinear local rotation | 11/13 useful |
| Bicubic local rotation | 12/13 useful |
| Feathered local rotation | 12/13 useful |
| Generated replacements | 0/26 rotation flags |
| Untouched, global rotation, translation, quantized translation, blur, tone, resize and 8-bit | 0/104 rotation flags |

All 35 flagged rotations covered at least half of the actual changed
pixels and reached at least 0.30 intersection over union. Of the 130
comparison deliveries, 34 reached a candidate assessment under the
rotation method. A missing candidate gives no evidence about whether
the picture was edited. All 13 8-bit deliveries were outside scope.
The serving detector reproduced all 169 recorded scalar rows.
[Exact scalar ledger](evaluation-panels/image-native-rotation-location-2026-11-13.json)

## Local use and replay

Use `VERITAS_IMAGE_NATIVE_INTERPOLATION_REVIEW=true` and
`VERITAS_IMAGE_NATIVE_ROTATION_REVIEW=true` on the local development API.
The optional extension uses its own evidence version, and the offline
verifier can recompute its exact crop and the observation from a signed
export and the submitted original.

Four held inputs completed isolated Full analysis after simulated payment.
The browser, Standard PDF and JSON agreed on the local rotation, the
untouched and generated controls, and the unsupported 8-bit delivery.
All four signed packages replayed against their originals. The reported
crop matched the native 16-bit samples.

```sh
python3 -B -IS backend/scripts/review_image_native_rotation_location.py \
  docs/evaluation-panels/image-native-rotation-location-2026-11-13.json \
  docs/evaluation-panels/image-native-stencil-interpolation-cc0-roster-2026-11-13.json
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze score replay serving-parity; do
  backend/.venv/bin/python -B -m scripts.image_native_rotation_location \
    "$phase" "$run" "$earlier_stencil_run" "$rawnind" "$pixls" || exit 1
done
```
