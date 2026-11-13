# Cross-scale gradient dependence around small JPEG edits

This CPU study measures whether a local image edit changes how fine and coarse directional gradients occur together at the same source pixels. Natural-image statistics show dependence between nearby wavelet scales. Recent JPEG inpainting research motivates retaining lower frequencies as well. This implementation uses source-pixel gradient relationships, no outside code, and no trained weights. [Primary cross-scale statistics study](https://www.cns.nyu.edu/pub/lcv/buccigrossi97-reprint.pdf) · [JPEG inpainting study](https://pubmed.ncbi.nlm.nih.gov/42335053/) · [Earlier residual sign comparison](image-residual-sign-transfer-2026-11-11.md)

On each complete native 16 × 16 cell, measure scale-normalized horizontal and vertical luma gradients at Gaussian scales of one, two and four pixels. Each cell contributes 64 source positions. For each adjacent scale pair, measure an estimate of Kendall tau-b on the gradient magnitudes from 192 seeded sample pairs. Measure centered sign correlation separately. Shift the same sampled coarse grid cyclically by two or three positions in either axis to form four controls. This preserves the exact sampled marginals. Subtract the average shifted response to isolate dependence tied to alignment between scales. Require enough local gradient activity and reject heavily clipped cells. Independently drawn responses had mean contrasts close to zero, identical responses had a mean contrast close to one, and flat images produced no proposal.

Rank native rectangles by a coupling deficit relative to a surrounding two-cell ring, after scaling by the ring spread. Both adjacent scale pairs must support a candidate. The agreement arm uses the weaker amplitude and polarity response from one common orientation. Separate fixed operating points are based on the second-highest source maximum across 12 ordinary-only calibration photos per quality and arm. The scope, coordinates, source commitments and code were fixed before held scoring. The incumbent comparator is the existing aligned luma check on the same JPEG bytes.

## Source-separated evaluation

Two retained source cohorts contain 12 and 16 distinct attributed photographs, with 14 LaMa removals and 14 Stable Diffusion 1.5 insertions in total. Their Flickr photos, photographers, encoded and decoded original images are separate from calibration and from each other. All had previous project exposure under other methods. Individual originals carry CC BY 2.0 and publisher contributions CC BY 4.0. The release identifies Apache 2.0 LaMa and CreativeML OpenRAIL-M Stable Diffusion 1.5. No generators were run. [Publisher rights](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution) · [Earlier and later source qualification](image-jpeg-contour-gap-transfer-2026-11-12.md)

Each source has sharp and inward-feathered edit variants, exact masks for changed pixels before final JPEG encoding, and same-site ordinary blur, color, copy and annotation controls. Other ordinary controls include the original, global resize, earlier compression, and publisher-matched local or global autoencoder processing where available. Every variant goes through the same 4:2:0 delivery path at JPEG75 and JPEG90. Useful localization needs 30% intersection over union and 50% recall of exactly changed pixels. The advancement rule requires three added useful feathered source groups across both families, at most two additional ordinary groups and at most four wrong edit locations per quality and arm.

## Held results

| Delivery | Method | Useful sharp / feathered, out of 28 each | Useful before the point, out of 56 | Ordinary flags /212 | Wrong edit regions |
| --- | --- | ---: | ---: | ---: | ---: |
| JPEG75 | Current luma | 0 / 0 | 0 | 1 | 0 |
| JPEG75 | Amplitude dependence | 0 / 0 | 1 | 0 | 0 |
| JPEG75 | Polarity dependence | 0 / 0 | 0 | 14 | 6 |
| JPEG75 | Amplitude and polarity agreement | 0 / 0 | 0 | 6 | 2 |
| JPEG90 | Current luma | 0 / 0 | 0 | 0 | 0 |
| JPEG90 | Amplitude dependence | 0 / 0 | 1 | 0 | 0 |
| JPEG90 | Polarity dependence | 0 / 0 | 0 | 12 | 4 |
| JPEG90 | Amplitude and polarity agreement | 0 / 0 | 0 | 5 | 0 |

All 536 held deliveries produced a candidate rectangle in every new arm. The amplitude arm was quiet on ordinary files but almost always put a different region first even before the review point. Polarity highlighted ordinary views from five source groups at JPEG75 and four at JPEG90. The agreement rule highlighted ordinary views from four and three source groups, respectively. Neither amplitude nor polarity met the frozen advancement rule. The check stays offline and contributes zero confidence weight.

## Local replay

The CPU run completed in 92.695 seconds. The local readback regenerated 704 delivery and exact mask bindings and independently checked 2,648 native overlap measurements. Raw photos, crops and per-file rows remain local. [Aggregate and replay commitments](evaluation-panels/image-jpeg-interband-dependence-2026-11-12.json)

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_jpeg_interband_dependence "$phase" \
    --archive "$archive" --encoder "$prior_encoder" --earlier "$earlier_f5" \
    --later "$later_f5" --output "$run" --publication "$receipt" || exit 1
done
```
