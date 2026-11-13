# Frozen JPEG patch heads on new source photographs

The earlier matched-head probe compared a binary head with one that gave ordinary autoencoder processing a separate label. Both found three softly blended edits from one editor family on previously used photographs. This transfer freezes the exact heads, encoder, rectangle search and score points. It fits and recalibrates nothing. [Earlier probe and model terms](image-three-class-patch-probe-2026-11-12.md)

## What the new panel measures

The frozen roster targets six Kandinsky and six Stable Diffusion 1.5 photo groups. Per-photo grants are CC BY 2.0 and the publisher contributions carry CC BY 4.0. The dataset pins the Apache-2.0 Kandinsky inpainting checkpoint and CreativeML OpenRAIL-M Stable Diffusion 1.5 inpainting checkpoint. The original, edited, processing and intent-mask PNGs all reproduce their publisher commitments at the same dataset revision. The test uses no editing checkpoint, published detection head or customer upload. [Publisher data and attribution terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution) · [Kandinsky checkpoint](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint/tree/db790ad5cbcabed886f069ef2710774657621702) · [Stable Diffusion checkpoint](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/tree/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb)

Exclude the prior probe's entire fit, calibration and held banks, the earlier regional exposure graph, the frozen residual panel, the two F5 research panels and additional qualified photo records referenced by those runs. Check source identity, individual photo ID, photographer profile and credited name, exact encoded and decoded content, and informative near thumbnails. Independence here means separation from the retained, recorded local exposure and fit banks. Both families come from the same publisher, both were seen during fitting, and the general encoder's pretraining exposure is unknown.

For Kandinsky, copy a fixed source-coordinate rectangle entirely inside the published generated background mask, using the same score-blind rule as the prior study. All six resulting edits affect 6.25% of their native pixels. Stable Diffusion supplies six original publisher insertions with exact native changes inside the semantic mask, covering 1.73% to 4.57%. For each, compute a sharp and an eight-pixel feathered edit using the exact same source placement. Match each source to its untouched original, same-site and full-frame autoencoder reconstruction, local blur, copy and tone. All 96 files use the same JPEG75 recipe. These constructed variants test whether a detector localizes known changes under the specified delivery and control conditions. Editor and edit-construction effects cannot be separated on this panel.

Require at least 30% intersection over union and 50% changed-pixel recovery. Advancement requires three new useful feathered regions across both families, at most two additional ordinary groups and at most four misplaced edited-file flags. The highest-ranked rectangle is also measured before applying its score point. Separately compare the median response at tokens with at least 50% exact edit occupancy against the same locations in matched controls. This second diagnostic knows the target and is unavailable for screening an unpaired upload.

## Local result

| Check | Useful sharp edits | Useful feathered edits | Ordinary files flagged | Ordinary source groups flagged | Misplaced edited-file flags |
|---|---:|---:|---:|---:|---:|
| Current JPEG luma | 0/12 | 0/12 | 0/72 | 0/12 | 0/24 |
| Frozen binary head | 2/12 | 2/12 | 22/72 | 5/12 | 8/24 |
| Frozen three-class head | 2/12 | 2/12 | 18/72 | 5/12 | 6/24 |

The two learned heads found the same two Stable Diffusion insertions and missed all six Kandinsky patches. The three-class head reduced ordinary flags and misplaced edited-file flags, while both still flagged at least one ordinary file in five photo groups. Both fell short of all three advancement requirements. Before applying the point, each head's top feathered box met the overlap criterion on three Stable Diffusion cases and zero Kandinsky cases. The three-class head put the feathered and original rectangles at the exact same location in seven of twelve groups. The binary head did so in six.

At known edit coordinates, both heads scored the feathered edit above the original, same-site autoencoder view, blur and tone controls in 6/6 Kandinsky groups and 5/6 Stable Diffusion groups for each comparison. The same-photo copy comparison was weaker, at 5/6 and 4/6 respectively. A measurable response at the known edit did not transfer to reliable blind localization. The app's screening and confidence remain unchanged.

## Local replay

Twelve groups passed intake on the first twelve attempted matches. Three frozen reserves remained unopened. All 48 downloaded PNGs totaled 23.3 MB. The one-thread offline model stage processed 96 deliveries in 1,248 tiles across 312 batches in 42.73 seconds. Reported process peak RSS was 858.2 MiB. A separate no-model readback reconstructed 96 exact JPEG and mask bindings and their known-site token support, checked 288 source rectangles and flags, and reran the 96 current JPEG comparisons. [Runner](../backend/scripts/image_three_class_patch_fresh.py) · [Aggregate and audit](evaluation-panels/image-three-class-patch-fresh-2026-11-12.json)

Supply the pinned BG-REAL metadata archive, exact prior matched-head directory, earlier residual and F5 directories, and the original pinned encoder. Use a fresh run directory. Only the acquisition stage needs public network access. The retained source and measurement files remain local.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in plan acquire run review; do
  backend/.venv/bin/python -B -m scripts.image_three_class_patch_fresh "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output "$run" --publication "$receipt" || exit 1
done
```
