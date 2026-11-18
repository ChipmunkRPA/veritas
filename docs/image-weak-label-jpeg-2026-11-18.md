# Whole-image supervision for local JPEG edit evidence

Three fitted heads yielded zero useful locations on 64 held JPEG deliveries
with staged localized edits. The existing luminance check also located zero.
The appearance-only head flagged 52 of 160 ordinary deliveries. Reconstruction
alone cleared the ordinary controls, and its highest-ranked box overlapped
enough of the edit in three deliveries before applying the fitted cutoff.
The experiment made no changes to the app.

## Method

The study tests a mechanism from recent weakly supervised localization
research. A model learns which whole images contain edits through a pooled
local response. Here the samples are paired by source and JPEG history.
An edited image with its original surroundings is compared with the untouched
photo. The same edited pixels with reconstructed surroundings are compared
with a whole-photo reconstruction. This tests whether the fitted response
can separate a local edit from an image-wide reconstruction. [Weakly supervised localization research](https://arxiv.org/abs/2609.36882)
· [Inpainting Exchange](https://arxiv.org/abs/2602.00192)

The code fits separate lightweight heads using frozen appearance features,
exact-grid reconstruction residuals, and their combination. The appearance
encoder is Apache-2.0 and the reconstructor is MIT. Each head pools scores over
two-by-two neighborhoods of 14-pixel cells and trains for 75 iterations using
whole-image labels. Edit masks are used to evaluate proposals. This study
uses the previously published source panel, including 16 fit, eight calibration
and 16 held groups separated by source. The same frozen edits and ordinary
JPEG controls are reused across heads. [Panel and earlier comparison](image-vae-double-jpeg-2026-11-18.md)
· [Appearance checkpoint](https://huggingface.co/facebook/dinov2-small)
· [Reconstruction checkpoint at the tested revision](https://huggingface.co/madebyollin/taesd/tree/9565f9356505d7c3537d0e15c558520ed95142ee)

Each delivery profile uses a cutoff strictly above the second-highest ordinary
source maximum in calibration. A useful region needs at least 0.30 intersection
over union and 0.50 edited-pixel recall. The baseline came from the earlier
experiment and was joined only after all 224 held encoded-byte hashes and case
identities matched.

| Held delivery | Head | Correct locations / 32 | Misplaced edit flags | Ordinary flags / 80 |
| --- | --- | ---: | ---: | ---: |
| JPEG 60 | Appearance | 0 | 12 | 28 |
| JPEG 60 | Reconstruction | 0 | 0 | 0 |
| JPEG 60 | Joint | 0 | 4 | 0 |
| JPEG 90 → 60 | Appearance | 0 | 17 | 24 |
| JPEG 90 → 60 | Reconstruction | 0 | 0 | 0 |
| JPEG 90 → 60 | Joint | 0 | 4 | 0 |

The dependency-free reviewer recomputes the six cutoffs and tallies from the
published numeric receipt. The original runner can replay the fit and held
inference with the model assets and archived panel inputs. The published
receipt contains source-group numbers, scores, and overlap metrics.
[Case-level results](evaluation-panels/image-weak-label-jpeg-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_image_weak_label_reconstruction \
  docs/evaluation-panels/image-weak-label-jpeg-2026-11-18.json.gz
```
