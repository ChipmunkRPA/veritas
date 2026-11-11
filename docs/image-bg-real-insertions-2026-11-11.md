# Localizing small insertions against matched processing controls

The earlier patch and crop comparisons missed the six FLUX Klein edits. This test holds those crop rules fixed and checks them on small Stable Diffusion 1.5 insertions from a new, rights-qualified source panel. It measures how often a clue also appears after an ordinary autoencoder round trip. [Earlier comparison](image-relative-patch-transfer-2026-11-11.md) · [Publisher methodology](https://arxiv.org/html/2607.26232v3)

## Source qualification

The locked roster selects base PNGs from the BG-REAL publisher test splits, four distinct source groups from each of the 2–5%, 5–10%, and 10–15% declared semantic-area bands. There were 109, 11, and zero eligible groups, respectively. The empty band contributed no cases. For the other two bands, all eight selected groups and all 32 source assets qualified. Each group has an attributed CC BY 2.0 photo, its insertion, its matched autoencoder control, and the publisher's binary edit mask. The publisher releases its edits and masks under CC BY 4.0 and records the CreativeML OpenRAIL-M insertion model revision. The study uses the permitted, published media as local evaluation data. No generator weights were acquired or run. [Dataset and attribution terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution) · [Pinned inpainting model and terms](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/blob/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb/README.md)

Before scoring, the intake checked author and photo identity against the exact earlier source inventory and compared decoded hashes and near-duplicate thumbnails against the 866-source graph and the 36 prior BG-REAL groups. Every edited source had nonempty exact RGB changes contained within the publisher mask. The exact changed pixels covered 3.07–5.61% of the source. The matched processing control has zero insertion truth, even though it changes many RGB values. This panel was previously unscored in the project. Upstream model training overlap is unmeasured.

## Local comparison

On PNG, compare the configured regional route against the previously frozen absolute and median-relative crop rules. Also evaluate a union of the current route and the relative crop. For JPEG75, evaluate only the two crop rules. The current app uses a separate JPEG route. The two encodings share source groups, and JPEG truth refers to the exact precompression insertion. Useful localization requires IoU of at least 0.30 and recall of at least 0.50.

| Format and check | Useful edit regions /8 | Unchanged originals flagged /8 | Processing controls flagged /8 |
| --- | ---: | ---: | ---: |
| PNG current regional check | 0 | 0 | 0 |
| PNG absolute crop | 4 | 2 | 8 |
| PNG relative crop | 4 | 0 | 4 |
| PNG current or relative crop | 4 | 0 | 4 |
| JPEG75 absolute crop | 0 | 2 | 8 |
| JPEG75 relative crop | 0 | 0 | 2 |

The relative crop added four correctly located PNG insertions with no unchanged-original flags. It also flagged four matched processing controls and lost all useful regions under JPEG75. Both crop rules remain experimental. The matched controls distinguish sensitivity to generative processing from useful localization of a semantic edit. A fresh matched comparison is needed to evaluate any control that suppresses the shared processing response.

The CPU run completed 24 current-route forwards and 1,200 crop inputs in 240 batches. All 24 current PNG masks matched a separate source-coordinate implementation. Integer pixel counts independently reproduced the metrics for all 48 media views and 144 arm measurements. The process exited zero in 70.5 seconds with approximately 1.16 GB peak RSS. Only aggregate results and commitments are published. [Intake and exact comparison counts](evaluation-panels/image-bg-real-insertions-2026-11-11.json)

## Separate holdout for shared processing artifacts

The follow-up kept the relative crop rule and selected region fixed. It also measured the median raw response of crops with no overlap with the selected region. This asks whether the rest of the image shows a broad processing response. The extra point was the second-lowest remote median among four flagged processing controls in the earlier panel. Passing requires a remote median strictly below that point. Fitting used only the eight earlier processing controls and occurred before any holdout scoring.

Eight other sources were selected with the same area bands and complete exclusion of the earlier panel's identities and exact or near-duplicate inputs. All 32 assets qualified. Exact insertions covered 2.25–5.53% of source pixels. On these native PNGs, the original relative rule localized 2/8 edits and flagged 3/8 processing controls. The additional remote requirement retained both useful locations and the third partial match. It cleared all three processing-control flags, and both rules flagged zero of eight unchanged originals. The 24 held-out views and eight fit views used 800 actual crop inputs in 160 CPU batches. An independent integer replay reproduced all 48 held-out arm outcomes. The run exited zero in 43.8 seconds at about 619 MB process RSS.

This improvement covers one published generator, one type of processing control, eight held-out sources and native PNG. JPEG transfer, other editing tools and app integration remain open. [Exact holdout and paired counts](evaluation-panels/image-crop-remote-context-2026-11-11.json)
