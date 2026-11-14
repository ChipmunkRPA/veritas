# Local signed relationships in image residuals

Three single-upload checks tested for changes in short-range relationships
within and across color channels. None found an additional correctly located
edit at the fixed review points. The app keeps its existing image checks.
[Frozen measurements](evaluation-panels/image-residual-third-order-transfer-2026-11-13.json.gz)

The runner subtracts a blurred version of luminance and two color differences,
then normalizes the signed remainder by local signal power. On an 8-pixel grid,
it compares products of two nearby positions, products of three positions, and
both sets together. A small linear fit uses the same source coordinates in 24
previously used TGIF/TGIF2 source groups and balances the contributions by
source. A disjoint set of 24 ordinary source groups sets the three review
points independently for each delivery format. A single rectangle is ranked
against its immediate surrounding ring. The source image and edit mask are
used to fit or score the experiment. They are absent from the input to the
localizer. Small edit size and generator differences motivated the test.
[Inpainting benchmark](https://arxiv.org/html/2512.16688) ·
[Research on color residuals](https://arxiv.org/html/2608.20558v1)

| Held source bank | Format | Sharp or feathered edits | Current physical comparison localized | Best candidate localized | Ordinary files per format | Extra ordinary flags across candidates |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TGIF/TGIF2, 48 source groups | PNG | 96 | 0 | 0 | 144 | 5–15 |
| TGIF/TGIF2 | JPEG75 | 96 | 0 | 0 | 144 | 7–13 |
| TGIF/TGIF2 | JPEG90 | 96 | 4 | 0 | 144 | 7–15 |
| BG-REAL, 12 source groups | Each of PNG, JPEG75 and JPEG90 | 24 each | 0 | 0 | 36 each | 0 |

A useful rectangle covers at least half the changed pixels and reaches 0.30
intersection over union with the exact pixel mask. Even without the review
points, the three-position feature alone never ranked a useful location in any
format. The joint fit ranked up to six useful locations on TGIF/TGIF2 and up
to three on BG-REAL in any one format. The PNG sharp-boundary comparison only
assessed 10 of 240 TGIF/TGIF2 inputs and 55 of 60 BG-REAL inputs within its
supported size. The JPEG reference is the aligned double-quantization check.
This therefore compares a new cue with format-specific physical measurements,
with no estimate of the full app's image accuracy. All controlled deliveries
share earlier project source material. No external weights or unpublished
photos are distributed.

Run `freeze`, `calibrate`, and `score` in order using the local archive and
the [BG-REAL one-source preparation](image-jpeg-signed-transitions-2026-11-12.md).
To recompute held arithmetic with the standard library only, decompress the
public ledger and run the reviewer.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze calibrate score; do
  backend/.venv/bin/python -B -m scripts.image_residual_third_order_transfer \
    "$phase" "$run_dir" --bg "$bg_plan_root" || exit 1
done
gzip -dc docs/evaluation-panels/image-residual-third-order-transfer-2026-11-13.json.gz > "$receipt"
python3 -B -S -m scripts.image_residual_third_order_transfer review "$receipt"
```
