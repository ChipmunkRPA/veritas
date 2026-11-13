# Prior JPEG chroma interpolation on small image edits

A prior JPEG color grid supplied one additional useful feathered edit location on 144 generated deliveries and 50 additional flags on 504 ordinary deliveries. The added location occurred in one quality-100 full-chroma panel. Both quality-95 deliveries gave zero useful top-ranked locations before applying the cutoff. The new check remains offline. [Audited results](evaluation-panels/image-chroma-interpolation-transfer-2026-11-12.json)

## Measurement

JPEG encoders can store color on a coarser grid than brightness. Common decoders interpolate missing color samples, which can leave different residual energy at even and odd pixels. This implementation measures that difference independently on each image axis and looks for a local interruption of the shared phase in the submitted JPEG. Earlier work studies a separate periodic color artifact from the encoder's rounding process and shows how editing and recompression can affect it. The implementation here uses the decoder's interpolation relationship and no external code or weights. [Decoder interpolation formula](https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/src/jdsample.c) · [Chroma-subsampling forensics](https://faui1-files.informatik.uni-erlangen.de/public/publications/mmsec/2019-Lorch-IFC.pdf)

For each complete native 16-pixel cell, measure the absolute four-tap residual with coefficients `[-1, 3, -3, 1]`, clipped at 24 chroma levels. Compare matched even- and odd-phase sums across both decoded chroma channels. Test one bounded rectangle from the frozen set. Each axis requires a signed contrast of at least 0.08 in at least three of four surrounding strips, separated from the rectangle's inner cells. The rectangle score uses the weaker axis's contrast drop. Independent arithmetic with exact quarter-weight interpolation confirmed that the four-tap filter vanishes in one phase and flips after a one-pixel shift.

## Sources and controls

Eight separate qualified photo groups provide seven ordinary control roles for point selection. Each operating point strictly exceeds the second-highest development source maximum, with a 0.06 floor. Two held panels supply 12 distinct photos each, with six SDXL and six Kandinsky edits per panel. Their underlying photographs had prior project exposure on other methods. Source groups are disjoint across the three roles under the retained publisher identity, attribution and near-duplicate qualification. The photo and dataset permissions are recorded in those intakes. No model code or weights are needed for this trial. [Earlier source qualification and rights](image-background-patch-transfer-2026-11-11.md) · [Later source qualification and rights](image-wavelet-region-transfer-2026-11-11.md)

Each held host first goes through a quality-90 4:2:0 JPEG round trip. A sharp or feathered rectangle replaces part of the decoded host with a licensed editor's output. The exact pre-final-encoding changed pixels define the evaluation target. Each source has a preserved host, a directly encoded photo without the constructed earlier grid, global tone and resize controls, and local blur, tone and autoencoder controls. The final encodings are quality 100 with 4:4:4 color, quality 95 with 4:4:4 color, and quality 95 with 4:2:0 color. Final 4:2:0 can introduce a new grid across the entire image. The detector sees one final JPEG and no earlier file. These are constructed compression histories, so the results apply to this test's prior and final encodings.

The comparator runs the app's unchanged aligned JPEG luma region check on the exact same delivered bytes. A useful region recovers at least half the changed pixels and reaches 0.30 intersection over union. Advancement needs at least two additional useful feathered regions in each panel of a delivery, with at most four new ordinary flags across at most three groups per panel. This study assigns zero AI confidence weight to the proposed measurement.

| Held delivery | Chroma useful sharp / feathered, out of 24 each | Luma useful sharp / feathered, out of 24 each | Chroma / luma ordinary flags, out of 168 | New ordinary flags from adding chroma |
| --- | ---: | ---: | ---: | ---: |
| Quality 100, 4:4:4 | 1 / 1 | 13 / 15 | 43 / 47 | 26 |
| Quality 95, 4:4:4 | 0 / 0 | 17 / 18 | 23 / 58 | 16 |
| Quality 95, 4:2:0 | 0 / 0 | 17 / 18 | 11 / 58 | 8 |

The quality-100 candidate assessed 125 of 216 deliveries. The quality-95 full-chroma variant assessed 36 of 216, and the 4:2:0 variant assessed 179 of 216. The quality-100 candidate flagged 10 of 24 preserved hosts and zero of 24 directly encoded originals. The luma check caught many of the planted edits and also frequently flagged conventional local blur or autoencoder processing. That response reflects a difference in compression history.

## Replay

From the repository root, set `development`, `earlier` and `later` to the retained qualified intakes and `run` to a fresh output path. Native readback independently checks each reconstructed JPEG digest, pre-final changed-pixel mask and recorded source rectangle overlap. The public record contains aggregate measurements and commitments. Source pixels and masks stay local.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze fit confirm; do
  backend/.venv/bin/python -B -m scripts.image_chroma_interpolation_transfer "$phase" \
    "$development" "$earlier" "$later" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_chroma_interpolation_transfer \
  "$earlier" "$later" "$run" "$publication"
```
