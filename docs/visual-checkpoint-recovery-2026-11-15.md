# Restored visual checks and heavy JPEG compression

The local supplementary image and sampled face-swap checks used model
files on Seagate. When the volume disconnected, both pinned checkpoints
were downloaded from their original releases and verified by byte count
and SHA-256. The new copies use the existing ignored local model paths.
Offline inference keeps the same weights and decision rules.
[Image checkpoint terms](image-smaug-v7-preparation.md)
· [Video checkpoint terms and held comparison](video-face-pe-confirmation-2026-11-11.md)

The image worker now checks both weight paths before attempting a
subprocess. A missing file or broken link yields the existing unavailable
status and no inspected-view crops. The main analysis can still run.

## Compression results

The earlier held panel supplies 16 ordinary photographs and 18 fully
generated images from three generator families. This comparison uses
all 34 earlier native source files and keeps their dimensions, colors,
and orientation when delivering at JPEG quality 35 and 15. The encodes
use standard 4:2:0 sampling. The live image worker reads the three same
types of source views and uses its existing CPU mean and fixed research
point. These are new deliveries of the same sources. This test measures
whole-image response after compression without measuring edited-pixel
localization or population error rates.
[Earlier held sources and reuse terms](image-whole-fresh-holdout-2026-11-12.md)

| Delivery | Generated images flagged | Ordinary photographs flagged |
| --- | ---: | ---: |
| Native, earlier measurement | 18/18 | 0/16 |
| JPEG 75, earlier measurement | 18/18 | 0/16 |
| JPEG 35 | 18/18 | 0/16 |
| JPEG 15 | 14/18 | 0/16 |

The four missed images at quality 15 include one from FLUX.2 max,
two from GLM-Image and one from Qwen-Image-2512. The app now adds a
short compression limitation when the check measures a no-flag result
on a file whose two JPEG tables exactly match a standard quality of 15
or lower. This limitation changes the description only. The public
interpretation does not expose internal model names or change AI confidence.
[Replayable JPEG decisions](evaluation-panels/image-whole-jpeg-transfer-2026-11-15.json)

## Paid app verification

Six local Full analyses used simulated USD 9.99 payments and queued no
unpaid jobs. The sampled face check flagged both matched publisher swaps
and returned no flag on the original clip. All three videos recorded the
same six source positions. Three original frames and face crops per
video matched the API assets and PDF. The ordinary photograph returned
no flag, while the generated JPEG showed three exact source views.
The known quality-15 miss retained its measured no-flag outcome with the
new compression explanation in JSON and the PDF. All six signed packages
verified against the original analysis input. These cases verify local
coverage and publication rather than broader accuracy.
[Local app receipt](evaluation-panels/visual-checkpoint-recovery-app-2026-11-15.json)

## Local replay

The numeric replay checks source commitments, matched counts, exact crop
coordinates and the frozen decision. It needs standard Python and
the checked-in JSON. Python 3.12 and 3.14 produced identical counts.

```sh
export PYTHONPATH=backend
module=scripts.image_whole_jpeg_transfer
receipt=docs/evaluation-panels/image-whole-jpeg-transfer-2026-11-15.json
python3 -B -S -m "$module" review "$receipt"
```

Use the earlier JPEG panel and pinned local research Python for native
inference. The app replay uses the exact earlier RTFS group 13 deliveries
and earlier generated and ordinary JPEGs. It holds source bytes by
digest, requires a simulated payment before the worker starts, checks
the API assets against the PDF and verifies signed packages from the
original file. The single compression example re-encodes one of the
observed misses and verifies its exact delivery digest before upload.
[RTFS source and permission review](video-common-source-preparation-v89.md)

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
tmp/image-smaug-v7-env/bin/python -B -m "$module" capture "$images" \
  models/research/image-v7-smaug "$run/jpeg.json"
backend/.venv/bin/python -B -m scripts.visual_checkpoint_recovery_app \
  "$run/paid" "$images" "$videos"
backend/.venv/bin/python -B -m scripts.visual_checkpoint_recovery_app \
  "$run/compressed" "$images" "$videos" --compression-receipt "$receipt"
```
