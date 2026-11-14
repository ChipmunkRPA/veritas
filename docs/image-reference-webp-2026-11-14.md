# Exact earlier-copy differences for lossless WebP

In the local app, open a completed image analysis and select **Compare with
an earlier copy**. If either file is an eligible opaque lossless WebP, the
other can be a compatible RGB PNG or an opaque lossless WebP. The comparison
shows the exact decoded pixel count, both source crops, and a mask of the
selected difference regions. The browser, comparison PDF and downloadable
JSON use the same native pixel coordinates. The JSON also records the source
file and decoded RGB hashes.

The check uses a single opaque VP8L image with 512–1,024 pixels on each side.
A PNG counterpart must have the same dimensions, use 8-bit RGB with no
transparency, and carry no color profile or orientation that changes the
comparison. Different RGB arrays require exact common samples across at
least 65% of the image and at least 12 detailed matching neighborhoods
distributed across the scene. A broadly different, re-framed, rotated or
incompatible pair can remain unassessed. Identical decoded pixels are
reported directly. Tiny disconnected differences are counted even when
they fall below the region display minimum.
[WebP container specification](https://developers.google.com/speed/webp/docs/riff_container)

## Local results

The source bank uses 26 attributed photos from a pinned BG-REAL release.
The two edited deliveries preserve the measured location through lossless
encoding. The unchanged, local tone, and two-pixel translation cases
measure three different expected behaviors. The result counts files with at
least 30% overlap over union and at least half the known changed pixels
covered. Exact means the connected-region mask contains every changed pixel.
All selected pixels were independently checked against the difference
between the two decoded source images.
[Benchmark and source rights](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

| Delivery | Comparable | Useful edit locations | Exact masks or control behavior |
| --- | ---: | ---: | --- |
| Sharp generated edits | 22/26 | 22/26 | 22 exact masks |
| Feathered generated edits | 22/26 | 22/26 | 21 exact masks |
| Unchanged lossless re-encodes | 26/26 | — | 26 exact matches, zero highlights |
| Ordinary local tone adjustments | 22/26 | — | 22 exact change masks |
| Two-pixel whole-image translations | 0/26 | — | 26 unassessed |

The comparison needs an earlier copy with enough shared detail. It
locates ordinary edits as well as generated edits and carries zero AI
confidence weight. These results measure paired evidence and do not
estimate single-upload detection quality.
[130-pair numeric receipt](evaluation-panels/image-reference-webp-transfer-2026-11-14.json)

The authenticated API tests verified the exact selected mask and both
PDF crops on a known change, and rejected an altered JSON count on replay.
The browser exercised an unsupported pair and an exact RGB match on a
locally paid test upload. A second run from newly downloaded publisher
sources reproduced the 130-pair receipt byte for byte.
[Browser and signed package receipt](evaluation-panels/image-regional-webp-app-2026-11-14.json)

## Replay

Review the published counts without media or detector weights. The
[PNG benchmark replay](image-edge-context-new-sources-2026-11-14.md#replay)
shows how to acquire the credited source assets.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.image_reference_webp_transfer review \
  docs/evaluation-panels/image-reference-webp-transfer-2026-11-14.json
backend/.venv/bin/python -B -m scripts.image_reference_webp_transfer measure "$result" \
  --sources "$photos" --receipt docs/evaluation-panels/image-edge-context-new-sources-2026-11-14.json
```

The [earlier-copy export reviewer](independent-testing.md#verify-an-earlier-copy-image-comparison)
recomputes the decoded counts, coordinates, crops and mask from both supplied files.
