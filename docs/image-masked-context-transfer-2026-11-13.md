# Context-only patch disagreement for small JPEG edits

This single-upload check measures how much a submitted patch’s visible representation differs from its representation when its pixels are hidden. The experiment tests whether surrounding image content gives the detector a new way to choose a region. A prior matched-head experiment found a response at known edit coordinates but usually ranked another region first. The new cue uses the existing Apache-2.0 patch encoder’s masking interface. It adds no trained weights and uses no earlier copy. Masked patch modeling and learned pixel dependencies motivate the intervention. [Masked patch pretraining](https://arxiv.org/html/2304.07193) · [Pixel-dependency study](https://arxiv.org/abs/2310.00234) · [Earlier localization result](image-three-class-patch-fresh-2026-11-12.md)

## What the test measures

Take the decoded JPEG75 pixels in unchanged native 224-pixel tiles, reflect padding only at the lower and right edges, and use the same 14-pixel token grid as the visible-token head. In one forward all pixels are visible. Four additional forwards hide a different quarter of the tokens each. Each token is hidden once, while its immediate horizontal and vertical neighbors are visible. The score is one minus the cosine between its normal final feature vector and the final feature vector produced with its pixels hidden. The pinned checkpoint has a zero-valued mask token. The synthetic intervention checks that changing the hidden patch changes its visible feature while the masked forward remains identical.

The primary check ranks native rectangles by localized excess contextual disagreement. The joint check averages that field and the frozen three-class response after separate median and median-absolute-deviation normalization. It uses fixed spread floors and the same rectangle geometry as the visible head. Each candidate’s score point is the second-highest source-level maximum among eight ordinary calibration groups. The existing three-class score point and aligned JPEG luma comparator stay fixed.

Eight calibration photos supply 40 ordinary controls. A distinct 12-photo confirmation set supplies 24 sharp or feathered edits and 72 ordinary controls. It contains six Kandinsky edits and six Stable Diffusion 1.5 edits, with matched unchanged photos, local and global autoencoder processing, blur, tone and same-photo copy controls. The project has already used these groups in other studies. The source attribution and exact masks come from the same pinned publisher deliveries as the earlier test. [Source lineage and panel limits](image-three-class-patch-fresh-2026-11-12.md) · [Encoder and Apache-2.0 terms](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)

Useful locations need 30% intersection over union and 50% recall of changed pixels. The continuation rule requires three additional useful feathered groups over current JPEG luma across both editor families, at most two added ordinary groups and at most four newly selected wrong locations. The frozen head is also compared directly. The experiment carries zero authorship confidence weight.

## Local results

| Check | Useful sharp /12 | Useful feathered /12 | Ordinary files flagged /72 | Ordinary groups flagged /12 | Wrong edited-file locations /24 |
|---|---:|---:|---:|---:|---:|
| Current JPEG luma | 0 | 0 | 0 | 0 | 0 |
| Frozen three-class head | 2 | 2 | 18 | 5 | 6 |
| Context disagreement alone | 0 | 0 | 11 | 3 | 2 |
| Joint context and three-class | 3 | 3 | 6 | 1 | 4 |

The joint check retained both useful Stable Diffusion 1.5 results from the frozen head and added the sharp and feathered versions of a third photo. The old head already placed that photo's top box correctly and its response fell below the fixed point. The joint check found the same three useful top boxes per edit type before applying either point. It improved which boxes were retained, with no new raw localization reach. All six Kandinsky edits still had wrong top locations or no retained region. Both candidates fell short of the frozen two-family rule and remain offline.

The synthetic input change left the masked forward numerically identical and changed the visible patch feature by 7.348 at its largest component. The Mac CPU run processed 136 JPEG deliveries, 1,728 tiles, 432 visible batches and 1,728 masked batches in 245.76 seconds. A separate no-model run reproduced 136 file and mask bindings, 504 native overlap and selection checks, both calibration points and all aggregates. [Audited aggregate](evaluation-panels/image-masked-context-transfer-2026-11-13.json)

## Replay

Use the exact prior matched-head run, retained photo archive and the same frozen transfer directory. The no-model readback reconstructs the JPEG and mask bindings, repeats the overlap arithmetic and recomputes both candidate score points from the calibration records. Source photos and per-file results remain local.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_masked_context_transfer "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --output "$run" --publication "$receipt" || exit 1
done
```

The standard-library reviewer checks the public stratum arithmetic and checks that recorded distinct source groups fall within bounds implied by the public file totals. It leaves private pixels and model computation unassessed.

```sh
python3 -B -I -S backend/scripts/review_image_masked_context_public.py \
  docs/evaluation-panels/image-masked-context-transfer-2026-11-13.json
```
