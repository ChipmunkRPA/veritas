# Exact small copies in lossless WebP

Chipmunk Verify can locate two crops in the same lossless WebP when their
RGB pixels match in their original orientation or after a quarter or half
turn. It checks the original submitted bytes and measures the differences
just outside both aligned areas. The browser, PDF, and JSON give the same
coordinates and crop references. The signed package verifies the measurement
against the original. The match leaves the direction of the copy and AI
involvement undetermined, with zero weight in the overall assessment.

The check runs as a fallback for upright, opaque, single-image lossless
WebP with 512–768 pixels on each side. It considers separated, textured
crops with 16–56 pixels on each side. Its decoder requires the simple
`VP8L` container. Alpha, animation, extended metadata, lossy encodes, and
other sizes are outside the profile. Existing geometry and trusted
Content Credentials take precedence. The implementation uses the existing
exact-pixel comparison and adds no model weights.
[WebP container specification](https://developers.google.com/speed/webp/docs/riff_container)
· [Original block-matching method](https://ws2.binghamton.edu/fridrich/Research/copymove.pdf)
· [The earlier PNG detector](image-exact-turned-copy-2026-11-15.md)

## Frozen comparison

Twenty-two camera photos and nine photos with repeated natural or architectural
patterns supplied 212 deliveries. The underlying photos appeared in earlier
studies. The small staged edits and lossless WebP deliveries were new. They
were made after resizing each JPEG-decoded source. All staged matches had to
recover both boxes with at least 50% intersection over union for each box.
The comparison includes the existing copy checks and the exact fallback.
[Camera and repeated-pattern source terms](image-small-affine-webp-2026-11-16.md#local-comparison)
· [BG-REAL source intake](image-masked-reconstruction-2026-11-17.md#reproduce)

| Staged edit | Existing checks | With exact fallback | Added locations |
| --- | ---: | ---: | ---: |
| Unrotated small copy | 1/22 | 18/22 | 17 |
| Quarter turn | 0/22 | 18/22 | 18 |
| Half turn | 0/22 | 20/22 | 20 |
| Feathered quarter turn | 1/22 | 19/22 | 18 |
| Total | 2/88 | 75/88 | 73 |

The added locations span 21 of 22 camera groups. The fallback added zero
misplaced regions and zero flags across 124 controls, including 31 mirrored-copy
deliveries used to challenge the non-mirrored check. The existing checks
flagged four of those 124 controls. Native verification recomputed every
exact match from its lossless delivered pixels. Five additional format controls
were outside the supported profile. The result covers these copy edits on this
profile. It gives no performance estimate for object removal, generated
images, or ordinary production uploads.

## Replay

The [frozen numeric receipt](evaluation-panels/image-exact-webp-copy-2026-11-18.json.gz)
records original and delivered hashes, both boxes for each staged edit, and
the measurements and matching decisions for every delivery. It contains
no photo bytes. Recount the results with the standard library.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.image_webp_exact_copy_transfer review \
  docs/evaluation-panels/image-exact-webp-copy-2026-11-18.json.gz
```

The same tool accepts `freeze` and `run` for a native replay with the three
source banks. Its `run` command retains four local proof files used by the
checkout replay, plus a fifth positive file available for spot checks.
From the backend directory, verify the selected inputs through simulated
checkout, report generation, and signed replay against the original.

```sh
.venv/bin/python -B -m scripts.image_webp_exact_copy_app "$held_directory" "$empty_app_output"
```

The local replay checked three matched deliveries, one ordinary lossless
file, and one lossy control. The quarter-turn file ran once with the
feature off and once with it on. Its overall assessment was identical.
All six jobs followed simulated payment. Five PDF and JSON exports
matched their signed packages and replayed against the originals.
The browser loaded both crops and marked their original positions on
each match. Both controls showed their coverage without crops.
[Local app replay receipt](evaluation-panels/image-exact-webp-copy-app-2026-11-18.json)

The development option is `VERITAS_IMAGE_LOSSLESS_WEBP_COPY_REVIEW=true`.
Reports already issued keep their recorded evidence version.
