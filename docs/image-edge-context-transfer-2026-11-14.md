# Checking whether image-region highlights depend on padding

The sensitive regional check fits the image without stretching and fills
unused model input with black. Earlier JPEG testing found fewer ordinary
highlights when the empty space extended the nearest source pixel. That
panel had no useful default locations, so it could not show whether the
change preserved correctly localized edits. This study keeps the same decoded
source pixels, fitted geometry, checkpoint, component selection and quality
mask. Only pixels outside the submitted image change.
[Earlier padding experiment](image-regional-padding-transfer-2026-11-14.md)
· [Original architecture and preprocessing](https://openreview.net/pdf/bd9565f0470fdc7df39503a2656625025b5d1820.pdf)

Compare the current black-fill pass, the edge-extended pass alone, and a
check that keeps a current region if at least half of its selected grid cells
are also selected by the edge-extended pass. The last method can drop a
misplaced region without changing the pixels of a retained region. It can
therefore improve overlap by removing irrelevant area. The two passes share
the same weights and provide no independent estimate of origin. An accurate
location covers at least half the exactly changed pixels and reaches at least
30% intersection over union.

## Sources

- Native PNGs reproduce all 236 uploads in the existing confirmation study.
  Fifty-nine photos in 48 connected groups each contribute one known edit,
  its untouched original, a local tone control and a local blur control.
  Input files, RGB values, truth masks and archived current responses are
  checked against their recorded hashes. [Native source and license record](native-regional-serving-v47-result.md)
- Twenty groups from the same PNG cohort supply a separate 40-delivery
  JPEG75 comparison. Each edited and untouched image was resized to
  768 pixels on the long side before JPEG encoding. The localization target
  uses the exact changed pixels after resizing and before encoding.
- Eight BG-REAL photo groups supply 32 additional JPEG75 uploads with LaMa
  removal, untouched, tone and blur views. The photos have recorded
  CC BY 2.0 attribution. These groups were absent from the prior padding and
  progressive JPEG pair tests and had previous exposure in other project
  evaluations. [Publisher source qualification and terms](image-bg-real-transport-v225-result.md)

## Local results

| Delivery | Measurement | Current | Edge-extended alone | Require support from both |
| --- | --- | ---: | ---: | ---: |
| Native PNG | Useful edit location | 45/59 | 45/59 | 46/59 |
| Native PNG | Misplaced highlights on edited files | 8/59 | 8/59 | 7/59 |
| Native PNG | Ordinary highlights | 34/177 | 29/177 | 26/177 |
| Native PNG | Source groups with any ordinary highlight | 12/48 | 11/48 | 10/48 |
| Re-encoded JPEG75 | Useful edit location | 2/20 | 2/20 | 2/20 |
| Re-encoded JPEG75 | Ordinary highlights | 3/20 | 4/20 | 3/20 |
| LaMa removal JPEG75 | Useful edit location | 0/8 | 0/8 | 0/8 |
| LaMa removal JPEG75 | Ordinary highlights | 2/24 | 3/24 | 2/24 |

The edge-extended pass alone gained one useful PNG and lost one of the 45
existing useful PNG locations. Requiring support preserved all 45. It made
one more result useful by removing an unrelated component, which reduced
the selected area from 79,328 to 77,456 pixels while preserving all 24,773
selected pixels inside the edit. It also withheld three untouched, three
tone and two blur highlights, with zero new ordinary flags. The 20 re-encoded
groups overlap the PNG cohort. The eight LaMa groups are separate, and none
of their edits received a candidate region under any treatment.

The support check is a candidate for a separate PNG transfer with new photo
groups and editors. This evaluation uses previously examined sources and
adds no new JPEG locations. Both padding variants stay outside serving.
The WebP format expansion has its own separate verification.

## Replay

The public receipt contains image and mask hashes, region pixel counts, and
exact overlap counts without pixels or model outputs that contain spatial
maps. The standard-library review rebuilds all aggregate counts. The native
run separately reconstructs each upload and exact truth mask before any
inference, verifies the 236 retained current responses and checks the actual
edge-extended input patches. It uses the pinned regional weights.

The first native attempt stopped after 276 rows when an evidence-module
import changed during the run. The rerun preloaded the complete source
readers, verified its implementation and inputs before inference, and
completed all 308 rows. All 276 earlier rows matched the rerun byte for
byte. A separate standard-library recount agreed with the stored measurements.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/image-edge-context-transfer-2026-11-14.json
python3 -B -S -m scripts.image_edge_context_transfer review "$receipt"
```

For a native rerun, pass the exact earlier plan and response index with
`--native`, the retained native source and edit qualification with `--pairs`,
the BG-REAL roster with `--background` and the existing private native maps
with `--raw-maps`. The first three paths default to the retained local
study folders. Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.

```sh
backend/.venv/bin/python -B -m scripts.image_edge_context_transfer freeze "$run"
backend/.venv/bin/python -B -m scripts.image_edge_context_transfer measure "$run" --raw-maps "$maps"
python3 -B -S -m scripts.image_edge_context_transfer review "$run/receipt.json"
```
