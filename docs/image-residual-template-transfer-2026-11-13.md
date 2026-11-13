# Recurring residual patterns in single-upload JPEGs

A single JPEG often gives little evidence for a small generated object removal.
The new check estimated a recurring pattern from the same upload and looked
for a local loss of that pattern. It located none of 50 known edits on 25 held
photographs. The strongest proposed region missed the edit on every positive
file, even before the cutoff. [Frozen results](evaluation-panels/image-residual-template-transfer-2026-11-13.json)

## Method

A recent paper studies whether a recurring residual pattern learned from a
single suspect image can provide a spatial reference. This test implements a
different version of that general idea with two spatially disjoint reference
folds, content matching and no model weights. The paper is distributed under
CC BY-NC-ND 4.0. Its source code, weights, figures and text were unused. The
previous Chipmunk Verify phase experiment measured excess periodicity inside
candidate areas. This one measures a deficit relative to independently
recurring patterns from elsewhere in the same upload.
[Single-image residual-pattern research](https://arxiv.org/html/2608.20558v1) ·
[Previous fixed-period experiment](image-phase-locked-residual-2026-11-12.md)

Subtract a fixed Gaussian smooth from luma and two color-difference planes.
For each complete 16 × 16 cell, retain signed residual means at fixed 2-,
4- and 8-pixel phases. Select distant peers with similar brightness, texture
and chroma from two separate spatial folds. A cell is eligible only if the
peers span three image quadrants and the two folds agree on a sufficiently
strong pattern. Compare the cell against each fold and retain the smaller
standardized shortfall. Search compact rectangles for clusters of supported
cells. This measures a repeatable pattern in the delivered image, which
can reflect compression or camera processing. It supplies zero AI-origin
confidence.

Eight separate photographs supply 48 ordinary calibration views. The strict
cutoff is the second-highest per-photo ordinary maximum, 4.9822673723 in
these score units. Twenty-five held sources provide one native LaMa object
removal, one feathered delivery and six controls each. The controls include
the original, matched local blur, tone and copy, an image-wide resize round
trip and a second JPEG encode. All files are ultimately saved as JPEG75
4:2:0. The test excludes source photo IDs, Flickr owners, credited authors and
original byte hashes across splits and two previously exposed single-upload
source ledgers. The selected archives had earlier project exposure under
other methods. Source photos have recorded CC BY 2.0 attribution. The
publisher separately documents CC BY 4.0 terms for its dataset contributions.
The replay verifies all original, edited, semantic-mask and exact-mask
commitments against the qualification records. It scores spatial overlap
against pre-encode changed pixels. A useful region requires at least 0.30
intersection over union and 0.50 changed-pixel recall.
[Primary source qualification](image-bg-real-transport-v225-result.md) ·
[Backup source qualification](image-bg-real-backup-intake-v262-result.md) ·
[Publisher attribution and terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

## Held results

| Check | Useful sharp | Useful feathered | Misplaced edit highlights | Ordinary highlights | Ordinary photo groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aligned JPEG luma | 0/25 | 0/25 | 0/50 | 5/150 | 5/25 |
| Residual template | 0/25 | 0/25 | 4/50 | 6/150 | 4/25 |
| Aligned luma plus template fallback | 0/25 | 0/25 | 4/50 | 10/150 | 8/25 |
| Strongest template region before cutoff | 0/25 | 0/25 | 50/50 | 150/150 | 25/25 |

The template had eligible cells on all 200 held deliveries. Its ordinary
highlights covered one original, two local blurs, one resize and two
intermediate JPEG encodes. The planted missing-phase smoke test passed.
Real-world edits in this panel did not produce a useful location with this
method. The comparison runs the existing aligned JPEG luma check. The
optional learned regional path was outside this test. The earlier learned
comparison separately reported zero useful locations on its six LaMa
sources. This probe remains offline.
[Separate learned regional comparison](image-regional-jpeg-transfer-2026-11-13.md)

## Reproduce

Run from the repository root with the retained qualification archives and
their cached assets. Supply the previous JPEG texture-rank and new-editor
exposure plans to exclude those sources. Use a fresh run directory. The
native review reproduced all 248 delivery and exact-mask hashes, boxes and
overlap measurements. The standard-library reader independently checked
the frozen row identities, cutoff and result arithmetic.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze calibrate held review; do
  backend/.venv/bin/python -B -m scripts.image_residual_template_transfer "$phase" \
    --archive "$v225" --archive "$v262" --excluded-plan "$jpeg_texture_plan" \
    --excluded-plan "$new_editor_plan" --output "$run" --receipt "$receipt" || exit 1
done
python3 -B -I -S backend/scripts/review_image_residual_template_transfer.py "$run" "$receipt"
```
