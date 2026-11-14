# Confirming the PNG repeat pass across editor families

The optional repeat pass kept all 11 useful locations from a single regional
inference and made one additional mask useful by removing an unrelated
component. It reduced ordinary highlights from **37 to 26 of 68** across
the two source banks. It is enabled for new analyses on the local development
instance. The app uses the same detector on the submitted PNG and its exact
horizontal reflection. A component stays visible when at least half of its
original grid cells overlap with the reflected mask after mapping back.
This measures repeatability under that transformation. It contributes zero
AI-authorship confidence. [Existing implementation and original experiment](image-regional-flip-transfer-2026-11-11.md)

## Held comparison

The study used 28 previously acquired photos and 108 PNG deliveries. The
TGIF/TGIF2 arm uses four photos per generator. These are the next
geometry-eligible held files after the first six in each generator family
used in the progressive JPEG replay. Each file has sharp and feathered
edited variants plus an untouched image, local blur and local color
adjustment. The BG-REAL arm uses sixteen other photos in two fixed
eight-photo source banks. Each has an untouched image, a Stable Diffusion 1.5
insertion and a whole-image autoencoder round trip. The exact changed-pixel
masks are committed before any inference. Every delivery satisfies the app's
768-pixel long side and at-least-512-pixel short side. The images had earlier
project exposure. This tests the existing rule on additional image families
and matched conditions, with no new model or cutoff fitting.
[TGIF/TGIF2 inputs](image-jpeg-signed-transitions-2026-11-12.md)
· [BG-REAL source rights](image-bg-real-insertions-2026-11-11.md)
· [Research on matched processing controls](https://arxiv.org/abs/2607.26232)

| Editor | Useful masks before → after | Ordinary highlights before → after |
| --- | ---: | ---: |
| Firefly | 0 → 0 of 8 | 6 → 6 of 12 |
| Stable Diffusion 2 | 2 → 2 of 8 | 7 → 2 of 12 |
| FLUX Fill | 4 → 4 of 8 | 3 → 3 of 12 |
| Stable Diffusion 1.5 | 5 → 6 of 16 | 21 → 15 of 32 |
| Total | 11 → 12 of 40 | 37 → 26 of 68 |

Highlights on untouched photos fell from 17 to 10 of 28. The reduction
across both banks met the frozen advancement criteria. The comparison still
missed 28 of 40 known edits, including all eight Firefly variants. On
autoencoder-only deliveries the check still highlighted 9 of 16. The repeat
does not expand an existing region or supply a new location where the first
pass finds no component. The 108 first-pass forwards triggered 64 repeats.
The data do not establish performance on other file formats, dimensions,
editors or natural upload distributions.
[Complete measurement ledger](evaluation-panels/image-regional-flip-confirmation-2026-11-13.json)

Two isolated simulated paid Full analyses reran the regional detector on a
previously evaluated edit and its untouched match. Both started with two
candidate regions. The edited file kept one; the untouched file kept none.
The PDF showed both withholding counts. JSON matched the evidence API,
and signed source replay recomputed the recorded overlap using the stored
maps and verified the crop against the supplied PNG. The local development
instance uses the repeat for eligible Full and Deep PNG analyses. The
repository keeps the setting opt-in as `VERITAS_IMAGE_REGION_FLIP_REVIEW=true`.
[Application verification receipt](evaluation-panels/image-regional-flip-current-app-2026-11-13.json)

## Reproduce

The scalar review needs only standard-library Python. For a native rerun,
use the cached TGIF/TGIF2 inputs and both original BG-REAL trial directories
from the source qualification above. The runner uses the existing
commercially permitted regional model and source-coordinate readout.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python3 -B -S -m scripts.image_regional_flip_confirmation review \
  docs/evaluation-panels/image-regional-flip-confirmation-2026-11-13.json
backend/.venv/bin/python -B -m scripts.image_regional_flip_confirmation freeze "$run" --bg "$insertion_trial" "$remote_trial"
backend/.venv/bin/python -B -m scripts.image_regional_flip_confirmation measure "$run" --bg "$insertion_trial" "$remote_trial"
```
