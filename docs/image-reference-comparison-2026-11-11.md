# Offline comparison with an earlier image

If you have an independently obtained earlier copy, you can compare it with the submitted image on your own machine. The command preserves the original source coordinates and writes a full-size difference mask, both original crops, and JSON with the source and artifact hashes.

```sh
cd backend
python -m app.cli compare-image earlier.png submitted.jpg --evidence-dir comparison
```

The destination must be a new directory. Omitting it prints JSON only. The two images need the same EXIF-oriented dimensions and measurable alignment. The tool refuses a global shift without moving the source pixels. It checks images up to about 16 megapixels. The displayed pixels indicate localized differences that survive the modeled processing. The earlier file's authenticity, the origin of an edit and the full chain of custody require separate evidence.

## Fixed comparison

The first comparison used native color differences after smoothing and a robust global tone fit. It located 12 of 12 feathered patches per format on its pilot, and also highlighted all 12 unchanged photos after JPEG75 delivery. The next comparison models the submitted JPEG's quantization and chroma sampling and adjusts large whole-image tone differences. On a separate twelve-photo panel it located 11 of 12 feathered patches in each format and removed all ten direct JPEG75 highlights from unchanged controls in that panel. It still highlighted ten of twelve JPEG75 files with an intermediate lossy encode.

The released local method keeps a local region when it also appears under at least two of three plausible earlier JPEG encodes. The frozen assumed qualities were 80, 88 and 96. The confirmation used eight other rights-qualified photo groups with publisher-derived Stable Diffusion 1.5 insertions, a newly feathered version inside each exact edit mask, and earlier qualities 83 and 92. The test has 128 views. Confirmation photographs were used in earlier unrelated project studies. They were outside both reference-comparison development panels. [Source rights and masks](image-bg-real-insertions-2026-11-11.md) · [Audited aggregate](evaluation-panels/image-reference-comparison-2026-11-11.json)

| Confirmation condition | PNG localized or highlighted | JPEG75 localized or highlighted |
| --- | ---: | ---: |
| Sharp generated edits | 8/8 | 8/8 |
| Feathered generated edits | 8/8 | 8/8 |
| Unchanged originals | 0/8 | 0/8 |
| Earlier lossy encode only | 1/16 | 5/16 |
| Whole-image contrast change | 2/8 | 2/8 |
| Whole-image autoencoder processing | 7/8 | 7/8 |
| Same-size three-pixel translation | 0/8 | 0/8 |

The translation controls were all unassessed. The previous one-delivery method highlighted 11/16 and 13/16 earlier-lossy-encode controls. A useful edit region needs at least 0.30 intersection-over-union and half the known changed pixels. An ordinary control counts if it has any displayed region. The two formats reuse the same source photos. Heavy processing can still create localized differences, and tiny or low-contrast changes may be missed. [Prospective runner](../backend/scripts/image_reference_compare_fresh.py)
