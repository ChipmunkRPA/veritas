# Earlier-copy comparison after a same-framing resize

The resized-reference check localized **17 of 24 feathered edits in both PNG and JPEG75**. It localized 17 of 24 sharp edits in PNG and 16 of 24 in JPEG75. The current integer-alignment check left all these resized inputs unassessed. The candidate passed the fixed research criteria in both formats. This trial measures an inspection cue for pixel differences with an earlier copy. It supplies no AI-origin or authenticity verdict. [Frozen protocol and audited aggregate](evaluation-panels/image-reference-resize-transfer-2026-11-11.json)

## Scope and construction

Two qualified panels supplied 12 source photos each. The two panels have distinct attributed photographs and photographers. Both panels had earlier project exposure on other methods. Each includes six SDXL and six Kandinsky publisher-derived patches. Each source supplies sharp and feathered versions of its known inserted patch, three ordinary controls, and three incompatible-framing controls. The 384 delivered views share 24 source photos across the two formats. The exact pre-encoding difference from a matched resized original defines each positive truth mask. [Earlier intake, image rights, and known patches](image-background-patch-transfer-2026-11-11.md) · [Later qualification and source separation](image-wavelet-region-transfer-2026-11-11.md)

The frozen fixture delivers images at 0.67× with OpenCV area resampling, 0.82× with Pillow bicubic resampling, or 1.17× with Pillow Lanczos resampling. Each slot has a fixed scale and resampler. Their effects therefore cannot be separated in this trial. The ordinary controls are resized originals, originals with an earlier JPEG83 encode, and whole-image contrast adjustments. The incompatible controls crop and resize, rotate by 2° and resize, or substitute an unrelated photo and resize.

Project the earlier RGB image onto the submitted grid with bicubic, bilinear, and Lanczos resampling. Require a proportional size change within 0.50× to 1.50×, minimum dimensions of 160 px, and a zero-pixel offset from the existing feature-and-tile alignment check on the bicubic projection. Compare each projected view with the existing tone-and-delivery-aware residual measurement. Select pixels only where at least two kernels support a region. The two crop rectangles use their respective native source grids. Earlier-copy edges round outward from the corresponding submitted region to retain the full area. [Existing alignment and delivery comparison](image-reference-comparison-2026-11-11.md) · [Resize runner](../backend/scripts/image_reference_resize_transfer.py)

## Results

| Source panel | Delivery | Sharp useful /12 | Feathered useful /12 | Resized originals assessed /12 | Ordinary controls highlighted /36 | Incompatible controls assessed /36 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Earlier | PNG | 7 | 7 | 9 | 2 | 0 |
| Earlier | JPEG75 | 6 | 7 | 9 | 2 | 0 |
| Later | PNG | 10 | 10 | 12 | 0 | 0 |
| Later | JPEG75 | 10 | 10 | 11 | 2 | 0 |

A useful location covers at least half the known changed pixels and reaches 0.30 intersection-over-union. The candidate highlighted six of 144 ordinary controls across formats. Three came from prior JPEG processing and three from whole-image contrast changes. It highlighted none of the 48 resized originals, 41 of which it could assess. All 144 incompatible-framing controls were unassessed. A framing refusal supplies no measurement of authenticity. The current comparison left all 384 resized views unassessed.

The predeclared decision requires six useful feathered locations, at least eight assessed resized originals, at most four ordinary control highlights including at most one resized original, and zero assessed incompatible controls per panel and format. All four cells passed. The candidate still missed 29 of 96 authored edit views and highlighted ordinary processing. This study covers proportional same-framing resizing at the tested settings. Combined crop and resize, rotation, unrelated sources, and application and export behavior require their own checks.

## Verification and replay

The CPU run scored 384 delivered inputs in 109.8 seconds without neural models. A separate local audit rebuilt all 384 encoded inputs and truth hashes from the retained assets, checked 85 native crop projections and 768 overlap records, and reproduced all counts and advancement decisions. A complete second detector run reproduced both case journals byte for byte, including the mask commitments. The aggregate keeps source and result commitments and excludes individual images, identities, coordinates, and results.

Run from the repository root with the exact two qualified intake directories and an unused output directory.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m scripts.image_reference_resize_transfer freeze "$earlier_intake" "$later_intake" "$output"
python -m scripts.image_reference_resize_transfer run "$earlier_intake" "$later_intake" "$output"
```
