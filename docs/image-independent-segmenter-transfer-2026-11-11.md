# Independent segmenter transfer to local image edits

The released segmenter flagged all 288 matched controls and localized none of 96 sharp or feathered edits. A separate local-contrast rule reduced the control flags to seven, with no useful localization. These results do not support adding the model to Chipmunk Verify. [Audited results](evaluation-panels/image-independent-segmenter-transfer-2026-11-11.json)

## Model and comparison

The checkpoint is the 107 MB DeepLabV3+ segmenter released by the AuthentiLens authors. The publisher declares the weights MIT and describes training on Stable Diffusion 2 and SDXL fully regenerated images. The evaluation used the pinned weights-only state, checked its exact SHA-256, and reconstructed the declared architecture using the MIT-licensed segmentation library. The publisher's separate application source was not imported. No classification override or image-level verdict was used. [Pinned model card](https://huggingface.co/iamgarvit/authentilens-weights/blob/f6cbed5c90a5b2a1f7919a1794ae50d1228ad0eb/README.md) · [Segmentation library license](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/LICENSE)

The first arm used the published 512 × 512 RGB preprocessing, sigmoid output, and quantized native-size mask at the published point. The second arm compared the mean response inside a 17-pixel neighbourhood with a surrounding 97-pixel neighbourhood and selected the highest-response connected region within fixed size bounds. Each format's local point was set on eight separate development photos using the second-largest source maximum over original, ordinary retouching and whole-image or local processing controls. The threshold-setting run used 96 checkpoint forwards, and both points were frozen before confirmation.

Two confirmation panels used 12 distinct previously qualified photo groups each, separate from development. Each paired PNG and JPEG75 used the same precompression edit mask. Controls include the original, ordinary blur and contrast, and local or whole-image autoencoder processing. The sharp and feathered insertions retain the source's original background outside the known edit. This tests transfer from training that includes fully regenerated backgrounds, motivated by evidence that some detectors respond to global reconstruction artifacts. The source photos had earlier project exposure. Exact overlap with the candidate's upstream training has not been established. [Original photo and generated patch qualification](image-background-patch-transfer-2026-11-11.md) · [Fresh source separation and blends](image-wavelet-region-transfer-2026-11-11.md) · [Inpainting Exchange study](https://arxiv.org/html/2602.00192v2)

| Panel | Delivery | Method | Sharp locations | Feathered locations | Controls flagged |
| --- | --- | --- | ---: | ---: | ---: |
| Earlier | PNG | Existing regional route | 0/12 | 0/12 | 11/72 |
| Earlier | PNG | Published mask | 0/12 | 0/12 | 72/72 |
| Earlier | PNG | Local contrast | 0/12 | 0/12 | 0/72 |
| Earlier | JPEG75 | Existing regional route | 1/12 | 0/12 | 7/72 |
| Earlier | JPEG75 | Published mask | 0/12 | 0/12 | 72/72 |
| Earlier | JPEG75 | Local contrast | 0/12 | 0/12 | 1/72 |
| Later | PNG | Published mask | 0/12 | 0/12 | 72/72 |
| Later | PNG | Local contrast | 0/12 | 0/12 | 6/72 |
| Later | JPEG75 | Published mask | 0/12 | 0/12 | 72/72 |
| Later | JPEG75 | Local contrast | 0/12 | 0/12 | 0/72 |

Useful localization required at least 0.30 intersection over union and 0.50 recall of changed pixels. The published mask highlighted a median 49–50% of control PNG pixels and 56–57% of control JPEG75 pixels, depending on the panel. This quantifies broad responses on these controls and does not estimate general deployment false-alarm rates. The earlier comparison reused 192 exact source-matched observations from the existing regional checks. The later panel evaluated the candidate independently. The single-thread CPU run made all 480 checkpoint forwards in 107 seconds. A separate audit reconstructed 384 encoded inputs and their truth hashes, checked the 192 matching baseline rows, and recomputed 960 overlap measurements from recorded integer counts. [Local runner](../backend/scripts/image_independent_segmenter_transfer.py)
