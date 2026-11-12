# Transfer of an existing crop control to additional image editors

The fixed remote-crop rule removed 13 ordinary PNG highlights and three ordinary JPEG75 highlights without losing a useful edit location relative to the same crop check. The remaining ordinary flags were high. This is a result on a candidate check, with no confidence contribution. [Audited transfer](evaluation-panels/image-remote-crop-editor-transfer-2026-11-11.json)

## Frozen comparison

The earlier relative-crop check selects the highest raw response among 25 overlapping crops and subtracts the median response. The additional requirement examines crops that do not overlap the selected crop. A high median in those remote crops can reflect image-wide processing. The prior Stable Diffusion 1.5 pilot chose both comparison points. This test kept them unchanged for both formats and used JPEG75 as a transport stress test. It uses the project's pinned MIT checkpoint. [Original rule and calibration](image-bg-real-insertions-2026-11-11.md#separate-holdout-for-shared-processing-artifacts) · [Pinned model](https://huggingface.co/OwensLab/commfor-model-384/tree/6076002bf0d9dd37537f965ee2f06f826c333b61)

The experiment used the same 31 held photo groups as the learned residual comparison. Seven newly qualified groups contain four SDXL and three Kandinsky edits and their source-matched ordinary controls. The other 24 groups had earlier project exposure. All three held panels are separate from the earlier control fit by the recorded photo IDs and author accounts. A useful location requires at least 0.30 native changed-pixel intersection over union and 0.50 recall. [Exact source qualification and masks](image-residual-patch-transfer-2026-11-11.md)

| Format and check | Sharp /31 | Feathered /31 | Ordinary highlights /186 | Untouched highlights /31 |
| --- | ---: | ---: | ---: | ---: |
| PNG relative crop | 11 | 5 | 70 | 6 |
| PNG with remote requirement | 11 | 5 | 57 | 6 |
| JPEG75 relative crop | 3 | 1 | 40 | 5 |
| JPEG75 with remote requirement | 3 | 1 | 37 | 5 |

All 16 removed highlights were on whole-image autoencoder controls. The seven fresh groups contributed four sharp and two feathered PNG locations, with seven ordinary highlights. Their JPEG75 deliveries contributed two sharp and one feathered location, with two ordinary highlights. The fresh JPEG positives account for three of the four useful JPEG locations across all 31 sources. That variation limits generalization even though the fresh subgroup showed useful evidence.

The first 384 views reused saved logits only after rebuilding and matching each exact crop hash. The 112 new views took 398 model batches, with 162 exact batch cache hits. An independent readback verified all 496 source images and truth masks, all 12,400 crop inputs, and 992 region measurements. It recalculated the remote median, flag arithmetic, and native overlap from saved logits. [Earlier exact crop ledger](image-crop-overlap-transfer-2026-11-11.md)

## Replay

Use the existing checkpoint, the prior remote-crop `points.json`, the crop-overlap run, both prior held intakes and the seven-group fresh intake. The renderer and offline verification keep image bytes and per-image responses local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze run verify; do
  backend/.venv/bin/python -B -m scripts.image_remote_crop_editor_transfer "$phase" \
    "$checkpoint" "$points" "$prior" "$earlier" "$later" "$fresh" "$run" || exit 1
done
```
