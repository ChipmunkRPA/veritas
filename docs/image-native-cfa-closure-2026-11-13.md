# Local interpolation consistency in camera-native image renders

The local Bayer phase-shift check located all eight conventional one-pixel
copy shifts in held 16-bit linear RGB crops. It highlighted none of the 40
ordinary views and declined all 24 unsupported or incompatible views. It
measures a change in how pixels are interpolated. An ordinary edit can
produce this result, and it carries zero AI-confidence weight.
[Exact replay receipt](evaluation-panels/image-native-cfa-closure-transfer-2026-11-13.json)

The check identifies a repeating Bayer interpolation pattern from two
disjoint spatial folds of one image. It measures how often red and blue
samples agree with the average of two neighbors at the expected pixel
phase. It then searches for a compact area where a different phase
continues to satisfy the same local interpolation relationship. This
compares spatial placement of the interpolated pixels directly. The
earlier experiment measured residual magnitudes in the green channel.
[Earlier camera phase cue](image-rawnind-local-edits-v244-result.md)

Four retained RawNIND originals and two new CC0 captures supplied six
development captures. A disjoint set of eight distinct camera models
supplied the held inputs. Ten newly acquired RAW files total 124,542,750
bytes. The source catalog records CC0 per file and every original was
verified against its published SHA-256. All scoring used a centered
2048 × 1536 native-grid crop, fixed linear RAW development and no
resize. The test regenerated the exact changed-pixel masks before
comparison. A useful region covered at least half the exact changed pixels
and reached 0.30 intersection over union. The 40 ordinary views contain
untouched renders, aligned conventional copies, local tone and blur, and
whole-image gain. The incompatible controls include 8-bit reduction,
resize round trips and adaptive demosaicing.
[New CC0 sources](evaluation-panels/image-native-cfa-cc0-roster-2026-11-13.json) ·
[Original publisher](https://raw.pixls.us/) ·
[RawNIND release](https://doi.org/10.14428/DVN/DEQCIM)

The separate phase-loss measurement located zero of eight classical
fills at its frozen point and highlighted three ordinary blur controls.
The phase-shift measurement also missed the eight fills. The panel
has no verified camera-native generated edits. The conventional shifts
were sharp, horizontally odd and 128 pixels across, so the measured
recovery does not establish performance on soft seams, small changes
or AI-generated replacements. The exact replay rebuilt all 80 held
views and matched their native pixel and mask commitments, locations
and scores.

## Replay

The source roster links to original publisher files. Save each original as
`sample-<id>.<lowercase extension>` in the RAW archive, and supply the
verified prior RawNIND cache containing the isolated decoder wheel. Run
the following phases using a fresh scratch directory.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze fit held review; do
  backend/.venv/bin/python -B -m scripts.image_native_cfa_closure_transfer \
    "$phase" "$rawnind" "$pixls" "$run" --receipt "$receipt" || exit 1
done
```
