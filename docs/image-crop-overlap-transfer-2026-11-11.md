# Estimating image edit locations from overlapping crop responses

Using overlap geometry reduced ordinary PNG highlights from **23/144 to 10/144** after removing broad score trends. It retained **1/48** useful PNG edit locations versus **5/48** for the strongest relative crop, and lost both useful feathered edits. In JPEG75 it increased ordinary highlights from **12/144 to 47/144** and lost the only useful edit location. The new readouts remain experimental.

## Fixed comparison

Research on overlapping patch predictions and separating local changes from background reconstruction motivates the design. This test uses a single existing MIT-licensed image classifier. It uses the published research to choose what to measure. It uses neither paper’s code, data, or weights. [DinoLizer](https://arxiv.org/html/2511.20722v2) · [DiffusionPrint](https://arxiv.org/html/2604.12443v1) · [Checkpoint](https://huggingface.co/OwensLab/commfor-model-384/tree/6076002bf0d9dd37537f965ee2f06f826c333b61)

Twenty-five overlapping crops each cover one third of the source width and height. The baseline selects the crop with the largest classifier logit and subtracts the median logit. Each new arm tests a native rectangle by calculating what fraction of that rectangle appears in each of the 25 crops. Candidate widths and heights use 4, 6, or 8 of 24 equally spaced source divisions. The search uses positions two divisions apart and requires that at least three crops intersect the rectangle and four exclude it. One arm removes a constant from this overlap pattern. The other also removes a linear trend in crop-center position. The score is the dot product of the measured logits with the unit-normalized residual overlap pattern. Each arm reports its strongest candidate.

The strict point for each arm and format is the second-highest of eight control-group maxima. Eight retained Stable Diffusion 1.5 source groups each supplied the original and an ordinary autoencoder processing control in both formats. Two separate groups of 12 held photos each supplied sharp and feathered insertions from SDXL or Kandinsky and six ordinary controls per format. All 32 credited photographers are distinct. The held photos had earlier project exposure. Truth is the exact changed-pixel mask before encoding. [Fit source qualification](image-bg-real-insertions-2026-11-11.md) · [First held intake](image-background-patch-transfer-2026-11-11.md) · [Second held intake](image-wavelet-region-transfer-2026-11-11.md)

A useful location requires at least 0.30 intersection over union and 0.50 recall of the exact changed pixels. The predeclared decision requires at least two newly localized feathered edits on distinct photos in each held panel, at most one lost useful baseline location, and at most two extra ordinary flags including at most one untouched flag per panel. The fit chooses a research point for each response. It supplies zero AI confidence weight.

## Held outcomes

| Format | Check | Useful sharp /24 | Useful feathered /24 | Ordinary flags /144 | Untouched flags /24 |
| --- | --- | ---: | ---: | ---: | ---: |
| PNG | Strongest relative crop | 3 | 2 | 23 | 2 |
| PNG | Overlap after centering | 0 | 0 | 5 | 0 |
| PNG | Overlap after removing position trend | 1 | 0 | 10 | 0 |
| JPEG75 | Strongest relative crop | 1 | 0 | 12 | 2 |
| JPEG75 | Overlap after centering | 0 | 0 | 21 | 3 |
| JPEG75 | Overlap after removing position trend | 0 | 0 | 47 | 7 |

Neither overlap arm added a useful feathered location. The baseline here uses the same crop logits and its own fixed control point. It is separate from the app's standard recompression and JPEG checks.

The CPU run finished in 369.9 seconds with 1,369 model batches of five crops. Another 711 batches reused exact matching input bytes. The offline reviewer rebuilds the 10,400 prepared crops and checks the independently implemented rectangular geometry, regression arithmetic, native overlaps, and fitted points. It uses saved model responses. [Frozen method and audited counts](evaluation-panels/image-crop-overlap-transfer-2026-11-11.json)

## Local replay

From `backend`, supply the qualified fitting and held intakes, the pinned checkpoint and an unused scratch directory. The input paths are the retained SD1.5 insertion trial, the earlier background-patch intake, and the later wavelet holdout. Keep source files and per-image responses local.

```sh
export PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.image_crop_overlap_transfer freeze "$fit" "$earlier" "$later" "$weights" "$run"
python -m scripts.image_crop_overlap_transfer run "$fit" "$earlier" "$later" "$weights" "$run"
python -m scripts.review_image_crop_overlap_transfer "$fit" "$earlier" "$later" "$weights" "$run"
```
