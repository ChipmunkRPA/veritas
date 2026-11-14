# Source-linked editing regions in lossless WebP

Chipmunk Verify now runs its sensitive regional editing check on eligible
lossless WebP stills when candidate image mode is selected. New Full and Deep
analyses can show suspected regions, the original crop, the selected mask and
the diagnostic overlay. The normal evidence view, Evidence API, JSON export
and Standard PDF keep the same source coordinates.

The check accepts the simple WebP container with a single opaque VP8L image,
512–1,024 pixels on each side, whose decoded format, frame count and dimensions
agree with its header. This keeps the decoded RGB pixels within the size of
the existing regional pass without resizing. Extended containers, animation,
alpha, lossy VP8 and other sizes are outside this expansion. WebP lossless
compression preserves the input pixels. A previously compressed image can
still be saved as lossless WebP, so the container cannot establish its history.
[WebP container specification](https://developers.google.com/speed/webp/docs/riff_container)
· [Lossless bitstream specification](https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification)

## Local verification

One already evaluated edit was uploaded as PNG and as lossless WebP.
The two files had different file hashes and exactly the same decoded RGB
pixels, model response and regional evidence. Both located the same region
at x 229, y 372, spanning 125 × 156 pixels. The original crop, source overview,
mask and overlay matched byte for byte. The browser displayed the checked
crop and mask, the report signature verified, and the frozen report evidence
matched the Evidence API and JSON. Both $9.99 checkouts used local test
doubles, and neither job existed before verified payment.
[Verification receipt](evaluation-panels/image-regional-webp-app-2026-11-14.json)

This is a file-format transfer of an existing editing check, with zero
contribution to AI authorship confidence. It uses one previous source and
does not estimate detection rates on WebP images encountered in the wild.
The regional check's existing weights and controls are described in the
[native image evaluation](native-regional-serving-v47-result.md).

## Replay

Use the PNG with the digest in the receipt, the installed regional weights,
and the local runtime with browser and PDF dependencies. The paid-flow test
encodes the WebP locally, starts an isolated test worker and compares both
uploads. Set `CHIPMUNK_REGIONAL_WEBP_FLOW_INPUT` to the PNG path. The receipt
includes file and exhibit hashes without private source media.

```sh
cd backend
.venv/bin/python -m pytest -q tests/test_native_regional_integration_v48.py -k webp_scope
.venv/bin/python -m pytest -o addopts='' -q \
  tests/test_native_regional_jpeg_paid.py::test_paid_lossless_webp_matches_png_regions
```
