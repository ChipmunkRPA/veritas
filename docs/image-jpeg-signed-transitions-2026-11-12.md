# Small generated edits in JPEGs with no earlier copy

Signed coefficient transitions found three additional useful feathered locations on the JPEG90 archive, one from each generator family. They also added ordinary highlights in seven source groups and gave seven wrong locations on edited views. A fresh BG-REAL transfer yielded no useful location at either JPEG quality. The cue remains experimental. [Frozen source, calibration and outcome commitments](evaluation-panels/image-jpeg-signed-transitions-2026-11-12.json)

## Measurement

The test reads quantized luma coefficients directly from the submitted JPEG. Each signed coefficient is clipped to five states, retaining negative, zero and positive values. One arm counts transitions between adjacent low-frequency coefficients inside each block. A second counts the same coefficient in horizontally and vertically adjacent blocks. The third combines both. A six-block local window pools the transitions. A regularized linear fit compares known edit pixels against the corresponding areas in ordinary photos, local blur and local color retouching. Source groups contribute equally. The highest compact rectangle must exceed a separate ordinary calibration point for its arm and quality. This is a different measurement from the existing JPEG double-compression check, which estimates whether a block lacks an earlier quantization trace. The implementation uses no generator weights or outside detector code. [Primary DCT transition research](https://www.sciencedirect.com/science/article/pii/S0031320312002440) · [Inpainting Exchange](https://arxiv.org/html/2602.00192v2)

The four archived TGIF/TGIF2 panels contribute 24 fitting source groups, 24 separate ordinary calibration groups and 48 held groups. The held views contain 16 examples each from Firefly, Stable Diffusion 2 and FLUX Fill. The source photographs had earlier project exposure. Fit, calibration and held source IDs and exact source hashes are distinct. The publisher distributes the original photo set under CC BY 4.0 and the edited release under CC BY-SA 4.0. The FLUX rows use published output pixels. [TGIF and TGIF2 release and terms](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea) · [FLUX Fill output terms](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/LICENSE.md)

Each held source has a sharp edit, an eight-pixel feathered edit, an untouched control, and matched ordinary blur and color adjustments applied at the same location. All views are encoded at JPEG75 and JPEG90 with 4:2:0 chroma sampling. The exact native ground truth comes from changed RGB pixels before common JPEG delivery. A useful rectangle covers at least half the exact changed pixels and reaches 0.30 intersection over union. Every selected region on a control counts as a flag.

| Delivery | Check | Sharp useful /48 | Feathered useful /48 | Ordinary flags /144 |
| --- | --- | ---: | ---: | ---: |
| JPEG75 | Existing compression cue | 0 | 0 | 0 |
| JPEG75 | Within-block transitions | 1 | 2 | 18 |
| JPEG75 | Between-block transitions | 1 | 1 | 14 |
| JPEG75 | Combined transitions | 1 | 1 | 17 |
| JPEG90 | Existing compression cue | 2 | 2 | 6 |
| JPEG90 | Within-block transitions | 3 | 4 | 21 |
| JPEG90 | Between-block transitions | 2 | 3 | 15 |
| JPEG90 | Combined transitions | 2 | 2 | 15 |

The frozen comparison required at least six added feathered locations across two families, at most three added ordinary source groups, and at most four wrong edit locations. All six quality/arm comparisons fell short. Multiple JPEG views of one photo count as one source group in the comparison. The score measures a departure in coefficient relationships. It supplies no authorship estimate.

## Fresh external challenge

Twelve newly acquired BG-REAL source groups provide six object insertions with Stable Diffusion 1.5 and six LaMa removals. Seven originals are from publisher test splits and five are from its training split, which was needed to obtain enough previously unused, rights-qualified LaMa sources. All 36 downloaded assets matched the pinned publisher revision and checksums. The original/edit pair had exact changes inside the published mask for every group. The 12 photos also passed encoded, RGB and near-thumbnail overlap checks against the project's prior attribution cache and one another. Source photos carry per-file CC BY 2.0 and the release carries CC BY 4.0 for publisher contributions. [BG-REAL release, rights and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

The same fitted weights, points and measurement were applied to all 120 fresh JPEG views. All three transition arms and the compression cue produced zero flags, covering 48 edited and 72 ordinary deliveries. The replay reproduced all input, mask and response commitments. No external benchmark material, model weights, or customer uploads are published in the receipt.

## Local replay

Use the retained TGIF/TGIF2 intake and the credited BG-REAL metadata archive. Use a new directory for each run and pass the v225 and v262 qualification records as `--exclude` in the BG-REAL stage. Only `acquire` uses network access to the pinned public release.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze calibrate held review; do
  backend/.venv/bin/python -m scripts.image_jpeg_signed_transitions "$stage" "$tgif_run" || exit 1
done
for stage in plan acquire freeze run review; do
  backend/.venv/bin/python -m scripts.image_jpeg_small_bg_real_holdout "$stage" \
    "$bg_run" "$bg_metadata" "$tgif_run" --exclude "$v225" "$v262" || exit 1
done
```
