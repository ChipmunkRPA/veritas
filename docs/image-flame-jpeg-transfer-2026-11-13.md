# Released FLAME on small JPEG edits

FLAME combines a learned local energy response with semantic segmentation.
The released detector gives us an independent comparison on the small
object removals that the app's native JPEG check and crop fallback missed.
The earlier project experiment measured a hand-built local energy cue.
This comparison runs the publisher's released trained state. [FLAME paper](https://arxiv.org/abs/2606.02178)
· [Previous energy trial](image-background-patch-transfer-2026-11-11.md)
· [JPEG tile trial](image-regional-tile-rescue-2026-11-13.md)

The publisher grants Apache-2.0 for the exact v0.1.0 FLAME weights and
configuration and for the pinned source revision. Meta grants Apache-2.0
for the separate SAM 2.1 base checkpoint. Together the two weight files
contain 330.9 MB. The runner checks each archive, state and configuration
digest before importing source. It extracts only the necessary model code
and configuration into temporary storage. The official SAM 2.1 builder
strictly restores its base state with `weights_only=True`. The separate
FLAME state also uses `weights_only=True` and covers all 100 intended
trainable tensors. Each tensor has the expected dtype and dimensions and
matches after assignment. [Exact FLAME release and license](https://github.com/phoenixnir/FLAME/releases/tag/v0.1.0)
· [SAM 2.1 checkpoint and license](https://github.com/facebookresearch/sam2#license)

## Method and held sources

Use the exact two-bank JPEG75 deliveries and source-pixel masks from the
tile trial. The first bank has 136 deliveries across 15 photos, including
seven LaMa removals and eight Stable Diffusion 1.5 insertions. This is the
development bank. The second bank has 64 deliveries from six other
LaMa photos and one shared photo with separate SDXL and Kandinsky edits.
It supplies the source-disjoint transfer. Controls include untouched
photos, local tone, blur and copy, resizing and earlier JPEG processing,
and where available local and global autoencoder processing. Both banks
had earlier project exposure through different methods. The earlier study
establishes disjoint photo, photographer and original-byte commitments
between banks. The upstream FLAME training inventory is separate and
complete pretraining membership is unknown. [First-bank source terms](image-jpeg-texture-rank-transfer-2026-11-12.md)
· [Second-bank source terms](image-masked-context-new-editors-2026-11-13.md)

The detector receives one decoded JPEG. Its source image is bilinearly
resized to 512 × 512 and mapped to unit RGB, matching the publisher's
evaluation path. Apply sigmoid to the mask logits, project the result
bilinearly to source coordinates, and take the two highest-ranked
connected components above the publisher's 0.5 pixel point. The comparison
uses the same JPEG reliability exclusions and minimum component area as
the local check. The second operating point is a strict exceedance of the
second-highest top-component response among the 15 ordinary-only
development group maxima. Freeze it before scoring the transfer. The
additive arm uses FLAME components only where the native local check
supplies no region. All comparisons use the same delivered bytes and the
exact pre-final-encode changed pixels. Useful localization requires 0.30
intersection over union and 0.50 recall.

The test asks whether the candidate finds at least two more useful
feathered LaMa locations on distinct photos while adding at most two
ordinary source groups and three misplaced edited-file highlights. This
is a prospective benchmark decision rule. The regional measurements
carry zero weight in AI authorship confidence.

## Results

| Panel | Check | Useful sharp edits | Useful feathered edits | Ordinary files highlighted | Ordinary photo groups highlighted | Misplaced edit highlights |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Development | Aligned JPEG luma | 0/15 | 0/15 | 0/106 | 0/15 | 0/30 |
| Development | Native regional | 5/15 | 5/15 | 8/106 | 3/15 | 2/30 |
| Development | FLAME at publisher pixel point | 1/15 | 0/15 | 102/106 | 15/15 | 29/30 |
| Held | Aligned JPEG luma | 0/8 | 0/8 | 3/48 | 3/7 | 0/16 |
| Held | Native regional | 1/8 | 0/8 | 5/48 | 2/7 | 2/16 |
| Held | FLAME at publisher pixel point | 0/8 | 1/8 | 38/48 | 7/7 | 11/16 |
| Held | FLAME at ordinary-control point | 0/8 | 0/8 | 0/48 | 0/7 | 0/16 |
| Held | Native regional with FLAME fallback | 1/8 | 0/8 | 5/48 | 2/7 | 2/16 |

The fixed FLAME readout located a sharp Stable Diffusion edit in
development and a feathered Kandinsky patch in the held bank. It
recovered no useful LaMa location across all 26 removal variants from
13 photographs. The ordinary-derived component point was
`0.889808714389801`. At that point FLAME selected no regions on any
of the 16 held edited or 48 ordinary deliveries. The additive arm
therefore returned the native check's existing evidence. This FLAME
configuration stays experimental. [Replayable outcome ledger](evaluation-panels/image-flame-jpeg-transfer-2026-11-13.json)

## Replay

The two-thread CPU run made 201 forwards across 200 JPEG deliveries,
including one bit-exact first-image repeat. The local replay re-encoded
all 200 source JPEGs, regenerated their exact changed-pixel truth, checked
728 packed masks and recalculated source-pixel overlaps and the ordinary
operating point. The public reader independently reproduces the point,
all localization and ordinary-highlight counts, and the advancement rule
from anonymous scalar measurements. It needs neither the images nor the
model weights.

```sh
python3 -B -IS backend/scripts/review_image_flame_jpeg_transfer.py \
  docs/evaluation-panels/image-flame-jpeg-transfer-2026-11-13.json
```

For a full CPU run, use the exact retained inputs and incumbent mask
journal from the [JPEG tile study](image-regional-tile-rescue-2026-11-13.md).
Put the five digest-pinned release assets named by the runner in a private
folder with the five pinned Python dependencies under `deps/`. The
official source archives use the released FLAME commit and Meta SAM 2
commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`. No model package is
installed into the app environment.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_flame_jpeg_transfer "$phase" \
    --assets "$assets" --tile-run "$tile_run" --archive "$archive" \
    --prior-probe "$probe" --residual "$residual" --early-f5 "$early_f5" \
    --later-f5 "$later_f5" --checkpoint "$checkpoint" --output-fresh "$fresh_transfer" \
    --external "$new_editor_transfer" --output "$run" --publication "$receipt" || exit 1
done
python3 -B -IS backend/scripts/review_image_flame_jpeg_transfer.py "$receipt" \
  --attach-local-ledger "$run/rows.json"
```
