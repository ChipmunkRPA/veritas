# Licensed reference-photo editing pilot

The pilot uses two CC BY 2.0 reference photos from the retained v39 cache. It
rechecks all 161 native JPEG records and their recorded attribution. The
overlap audit combines the v39 graph with the 36 BG-REAL originals and four
RawNIND captures. The later regional, JPEG and product-replay panels that reuse
these sources are covered through their source lineages. The checks use known
photo and creator identifiers, encoded and decoded pixels, and the fixed
thumbnail comparison. They can miss creator aliases, transformed duplicates,
unrecorded exposures and model pretraining overlap.

The procedure picks two distinct groups by a fixed salted source-file digest,
with one native 256×256 center crop per photo. No resizing or quality selection
is used. The two fixed prompts request square stickers in the center and at the
left edge. Each request uses seed 276004 and four steps. The worker passes the
photo as a reference and applies the 64×64 mask separately in source coordinates.
It saves the raw output and counts pixel changes both inside and outside that
mask. A qualifying composite preserves the original RGB exactly outside the
mask and has at least one exact change inside it.

Tone, blur and mirrored copy controls use the same source and mask boundary.
The original, controls, generator output, and observed change masks use lossless
PNG. The pilot measures construction and geometry. Semantic edit success,
detector accuracy, calibration and broader source transfer require separate
evaluation.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_flux_photo_preparation_v276 prepare
backend/.venv/bin/python -B -m scripts.image_flux_photo_preparation_v276 freeze
backend/.venv/bin/python -B -m scripts.image_flux_photo_preparation_v276 run
```

The freeze replays the source selection and control geometry before generation.
Original JPEGs and the acquired model remain on Seagate. Prepared photos,
controls, original photo credits, and generated PNGs remain in the private
project evaluation cache. The local v40 runtime runs offline under the same
6 GiB admission and sampled worker-tree RSS limits as the authored v271 trial.
Each source has one attempt with no source replacement or retry.
