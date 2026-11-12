# Regional image evidence after a reversible flip

The repeat pass retained all three useful locations found by sensitive screening and reduced ordinary flags from **109/144 to 68/144**. Untouched-photo flags fell from **18/24 to 10/24**. The recompression check flagged **23/144** ordinary inputs and localized none of the 48 known edits. The remaining false highlights limit the repeat pass to an opt-in local check. [Audited aggregate](evaluation-panels/image-regional-flip-transfer-2026-11-11.json)

## Method

Run the existing sensitive PNG localizer on the submitted pixels and on an exact horizontal reversal. Reverse the second retained mask back onto the original pixel grid. Compare the two individual masks, their shared pixels, and a rule that retains an original eight-connected component when at least half of its pixels are also selected in the reversed pass. The latter keeps the original component geometry. It uses no additional checkpoint, training, or calibrated authorship weight. A recent document image study used shifted tiling grids to remove spurious detections at patch boundaries. This experiment asks a similar stability question on photos, with a different transform and readout. [Recent forensic augmentation study](https://arxiv.org/html/2609.38391v1)

The two source-exposed panels contain 24 distinct photos by 24 credited photographers, with six SDXL and six Kandinsky source groups per panel. Each photo supplies a sharp and feathered known edit, its untouched original, and five ordinary processing controls. Exact changed pixels define the target. The controls include blur, contrast, and local or whole-image reconstruction. All 192 views use lossless PNG. The current recompression localizer ran independently on every original-orientation input. [Earlier source qualification](image-background-patch-transfer-2026-11-11.md) · [Later source qualification](image-wavelet-region-transfer-2026-11-11.md)

A useful region covers at least 50% of the known changed pixels with at least 0.30 IoU. The pre-run component rule requires 20% fewer ordinary flags than the sensitive localizer, at least 75% retention of its useful locations, and at least one additional useful location over recompression in each panel. These are evaluation criteria for this comparison. Both transformed passes use the same weights, so spatial agreement measures repeatability under this transform.

## Results

| Panel | Method | Sharp locations /12 | Feathered locations /12 | Ordinary flags /72 | Untouched flags /12 |
| --- | --- | ---: | ---: | ---: | ---: |
| Earlier | Sensitive screening | 1 | 0 | 47 | 8 |
| Earlier | Flipped pass mapped back | 1 | 0 | 41 | 6 |
| Earlier | Shared pixels | 0 | 0 | 38 | 6 |
| Earlier | Supported original components | 1 | 0 | 30 | 4 |
| Earlier | Recompression | 0 | 0 | 11 | 2 |
| Later | Sensitive screening | 1 | 1 | 62 | 10 |
| Later | Flipped pass mapped back | 1 | 1 | 68 | 11 |
| Later | Shared pixels | 1 | 1 | 54 | 9 |
| Later | Supported original components | 1 | 1 | 38 | 6 |
| Later | Recompression | 0 | 0 | 12 | 1 |

Both panels met the frozen comparative rule. Intersecting individual pixels lost one useful location. Keeping the original connected region when the repeat supported it preserved all three and removed 41 control flags. The controls reuse each source photo across treatments. General performance on other image sizes, formats, editors, and uploads needs separate testing.

The CPU run completed 576 forwards. The model-free auditor reproduced 384 input encodings and normalized input digests and 960 region measurements. Its overlap counts also matched 192 earlier measurements from the two original-orientation detectors on the later panel. The auditor checks the recorded selections and source inputs. It does not rerun the neural weights. [Earlier actual inference](image-active-png-transfer-2026-11-11.md)

## Local app

Set `VERITAS_IMAGE_REGION_MODE=candidate` and `VERITAS_IMAGE_REGION_FLIP_REVIEW=true` in development for Full and Deep analysis. The repeat covers PNGs whose oriented long side is 768 pixels and short side is at least 512 pixels. It runs only when the original pass produced a candidate. It checks the first two ranked components, preserves the supported components' original masks, and records exact pixel counts for the repeat. The evidence view and PDF explain how many regions were reviewed and withheld. JSON includes their source bounds and counts. The original response map remains available for inspection, and the repeat contributes zero authorship-confidence weight.

An isolated Full-profile test used the real app worker and installed regional checkpoint on a matched edited PNG and untouched PNG. Both original passes proposed two regions. The repeat retained one region on the edit and withheld both on the untouched image. With the repeat enabled or disabled, each fixture returned the same original regional score and overall decision. Simulated Stripe payment preceded every analysis. The evidence API and JSON matched the signed package, the PDFs included the repeat-pass findings, and the published crop matched the original pixels. The public exports used model-neutral wording. This test limited scheduling to the regional detector.

## Reproduce

From `backend`, use the rights-qualified retained intakes, the current pinned checkpoints, and an unused scratch output directory. Keep private masks and source images local. Provide the previous active PNG receipt directory to match the two original-orientation checks against their earlier inference.

```sh
export PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -B -m scripts.image_regional_flip_transfer freeze "$earlier" "$later" "$run"
.venv/bin/python -B -m scripts.image_regional_flip_transfer run "$earlier" "$later" "$run"
.venv/bin/python -B -m scripts.review_image_regional_flip_transfer "$run" "$earlier" "$later" --prior-later "$previous"
```
