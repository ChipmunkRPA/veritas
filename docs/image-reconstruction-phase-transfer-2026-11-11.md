# Reconstruction response for small edits without an earlier copy

All three reconstruction variants missed all 96 sharp or feathered positive deliveries. The highest-ranked rectangle failed the useful-location requirement on every positive, even before the operating point was applied. The primary variant also flagged 42 of 144 ordinary PNG deliveries and 23 of 144 JPEG75 deliveries. Both formats failed the advancement rule. [Audited aggregate](evaluation-panels/image-reconstruction-phase-transfer-2026-11-11.json)

## Method

Recent reconstruction-guided localization work measures how an autoencoder responds to edited image content. Inpainting Exchange shows why a whole-image detector can instead react to processing on the background. This study tests a small local response and compares it with ordinary autoencoder processing in the same image area. It uses a pinned MIT-licensed Tiny AutoEncoder checkpoint and the official Diffusers runtime. It uses no training or code from either research study. [Reconstruction-guided localization](https://arxiv.org/abs/2609.36882) · [Inpainting Exchange](https://arxiv.org/html/2602.00192v2) · [Pinned checkpoint](https://huggingface.co/madebyollin/taesd/tree/9565f9356505d7c3537d0e15c558520ed95142ee)

The decoder reconstructs the whole submitted image after four fixed offsets relative to its eight-pixel latent grid. Each input is reflect-padded to a multiple of eight. Cropping the output to the exact unpadded origin puts its RGB residual back on the same source grid. Each 16-pixel tile measures reconstruction error adjusted for its local detail. Very flat or clipped tiles are excluded. Candidate rectangles compare the inner tiles with a separated surrounding ring.

The fixed-phase arm uses the first reconstruction. The best-phase arm uses each tile's smallest error across all four reconstructions. The primary arm requires an error deficit in at least three phases at the same rectangle. This probes repeated local agreement without assigning authorship probability.

## Sources and rule

Eight development photos supply six ordinary controls each in PNG and JPEG75. The strict point per format and arm is the second-highest of eight source-level control maxima, with a minimum of zero. The points were written before either confirmation panel. The two 12-photo panels use disjoint photos and credited photographers from development and from one another. They had previous project exposure under other methods. Each includes six SDXL and six Kandinsky generated inserts, in sharp and feathered forms, plus original, blur, contrast and ordinary local or global autoencoder processing controls. Both encodings use the exact precompression changed-pixel footprint as truth. [Earlier source qualification](image-background-patch-transfer-2026-11-11.md) · [Later source qualification](image-wavelet-region-transfer-2026-11-11.md)

The comparison uses the local PNG sharp-boundary check and the existing native JPEG double-quantization region. Each useful region needs at least 0.30 intersection over union and 0.50 recall of the known changed pixels. To advance in a format, the primary arm must add at least two useful feathered locations per held panel while adding at most four ordinary flags across at most three sources per panel.

## Results

| Delivery | Check | Sharp locations | Feathered locations | Ordinary flags |
| --- | --- | ---: | ---: | ---: |
| PNG | Existing boundary | 12/24 | 0/24 | 14/144 |
| PNG | Fixed phase | 0/24 | 0/24 | 37/144 |
| PNG | Best phase | 0/24 | 0/24 | 43/144 |
| PNG | Three-phase agreement | 0/24 | 0/24 | 42/144 |
| JPEG75 | Existing native JPEG | 0/24 | 0/24 | 0/144 |
| JPEG75 | Fixed phase | 0/24 | 0/24 | 30/144 |
| JPEG75 | Best phase | 0/24 | 0/24 | 17/144 |
| JPEG75 | Three-phase agreement | 0/24 | 0/24 | 23/144 |

The primary arm selected the wrong area on 17 PNG and nine JPEG75 positives. It also flagged seven of 24 untouched PNG deliveries and four of 24 untouched JPEG75 deliveries. All views had enough measured tiles to generate a candidate rectangle. The miss occurred in ranking as well as classification. These results apply to this small reconstruction model and the fixed local readout on these inputs.

The Mac ran 1,920 encode/decode round trips in 338 seconds, including 384 development round trips. A separate Pillow replay checked all 384 encoded and truth commitments, recalculated 1,536 native overlap measurements and reproduced every outcome without running the model again.

## Reproduce

Use the checkpoint, config and license from the pinned publisher revision and the four isolated Python wheels whose hashes appear in the freeze. The Diffusers wheel requires Hugging Face Hub 1.32.0 in this scratch environment. The app's installed runtime is separate. From the repository root, provide the retained development and confirmation intakes. The audit rebuilds the delivered inputs and remeasures overlap without neural inference.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="backend:$runtime/lib" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_reconstruction_phase_transfer freeze \
  "$checkpoint" "$runtime" "$development" "$earlier" "$later" "$run" --device mps
backend/.venv/bin/python -B -m scripts.image_reconstruction_phase_transfer run \
  "$checkpoint" "$runtime" "$development" "$earlier" "$later" "$run" --device mps
backend/.venv/bin/python -B -m scripts.review_image_reconstruction_phase_transfer "$run" "$earlier" "$later"
```
