# Background-patch localization across two image editors

The crop check localized three of 12 generated patches in PNG, compared with zero for the current PNG route. Adding a check of distant crops removed four false highlights while retaining those three localizations. The combined crop rule still highlighted 23 of 48 controls in PNG and localized none of the same edits in JPEG75. Keep the current PNG and JPEG regional routes and retain the candidates for research.

## Panel and methods

We used six SDXL and six Kandinsky source groups from the pinned BG-REAL v2 generator holdout. The intake checked recorded source exposure and per-photo attribution, then selected a fixed rectangle fully inside each publisher edit mask. Copying only that rectangle from the publisher's edit changed 5.87–6.25% of native pixels and preserved the exact original pixels outside it. Each group supplied the generated patch, unchanged original, local and global autoencoder processing controls, and a local blur control. The same five views were encoded as lossless PNG and JPEG75. BG-REAL records CC BY 2.0 for the selected photos and CC BY 4.0 for publisher contributions. The source metadata pins OpenRAIL++ for SDXL Inpainting and Apache-2.0 for Kandinsky. [BG-REAL study and processing controls](https://arxiv.org/html/2607.26232v3) · [Dataset terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution) · [SDXL model card](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/blob/115134f363124c53c7d878647567d04daf26e41e/README.md) · [Kandinsky model card](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint)

The relative crop rule compares the strongest of 25 overlapping crops with their median response from a frozen MIT-licensed classifier. The distant-crop rule keeps that same region only when the median response in non-overlapping crops falls below its previously fixed point. The RGB energy rule looks for lower local variation than in the immediate surroundings, with separate operating points derived from eight prior source groups. INP-X motivates controlling for regenerated backgrounds. FLAME motivates testing local energy differences. Both ideas were tested with local implementations and existing weights. [Classifier and license](https://huggingface.co/OwensLab/commfor-model-384) · [INP-X](https://arxiv.org/html/2602.00192v2) · [FLAME](https://proceedings.mlr.press/v306/wang26iy.html)

## Results

A useful localization needed at least 0.30 intersection over union and 0.50 recall of the exact pre-encoding changed pixels. A control counted as highlighted whenever the method reported any region.

| Method | PNG useful edits /12 | PNG highlighted controls /48 | JPEG75 useful edits /12 | JPEG75 highlighted controls /48 |
| --- | ---: | ---: | ---: | ---: |
| Current regional routes | 0 | 7 | 1 | 4 |
| Relative crop | 3 | 27 | 0 | 9 |
| Relative crop with distant-crop check | 3 | 23 | 0 | 9 |
| Local RGB energy deficit | 0 | 14 | 0 | 12 |

The three PNG localizations included two SDXL edits and one Kandinsky edit. The distant-crop check suppressed four PNG highlights, all on global autoencoder controls. It suppressed none in JPEG75. The 23 PNG control highlights came from 11 of 12 source groups, so the added PNG sensitivity carried a broad false-highlight cost. The two formats reuse the same 12 photos and the patch shape was prescribed by the experiment.

A separate local audit reconstructed all 12 truth masks, reproduced all 120 encoded input digests, matched the 60 JPEG reference inputs by SHA-256, and checked the native-pixel arithmetic of 480 region measurements. The CPU stage processed 3,000 crop inputs and 60 current PNG forwards. The Metal stage made 60 current JPEG forwards. The aggregate includes results by generator, format, method, and control type. [Audited aggregate](evaluation-panels/image-background-patch-transfer-2026-11-11.json)
