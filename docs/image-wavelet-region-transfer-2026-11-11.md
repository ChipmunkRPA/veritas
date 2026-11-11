# Fine-detail contrast on blended local image edits

On the earlier paired panel, a simple fine-detail check localized one of 12 feathered edits in each format. It highlighted one of 72 PNG controls and two of 72 JPEG75 controls. On 12 additional source photos, neither method localized a sharp or feathered edit in either format. The current image checks remain in service. [Audited results](evaluation-panels/image-wavelet-region-transfer-2026-11-11.json)

## Method and source separation

The November revision of Inpainting Exchange tests how detectors respond when original pixels outside an edit are restored. Its wavelet analysis motivated a local test of relative detail rather than relying on an image-wide response. DiffusionPrint learns local generator fingerprints, but its official repository still lists its pretrained checkpoints as pending. FUSED combines forensic and semantic features and reports a substantial drop when the background artifact is removed. This study implements a separate small CPU method with no external training code or weights. [Inpainting Exchange revision](https://arxiv.org/html/2602.00192v2) · [DiffusionPrint repository](https://github.com/mever-team/diffusionprint) · [FUSED study](https://arxiv.org/html/2608.28302v1)

On each 16-pixel native grid block, the check measures three scales of Haar detail in luma and chroma. The proposals compare the interior of a rectangle with a separated ring of surrounding pixels. The simple arm looks for relatively lower fine luma detail. The fitted arm uses six luma, chroma and directional ratios and a regularized linear combination. Each format uses 96 paired positive–control measurements from eight Stable Diffusion 1.5 development sources. Eight other photos supply the four fixed review points. A proposal passes when its response strictly exceeds the second-highest source maximum for that method and format. The controls include ordinary blur, contrast, sharpening and autoencoder processing.

The paired test uses the same 192 encoded views as the earlier blended-boundary experiment. The prospective holdout excludes the fit, calibration, prior comparison and historical source groups using attributed photo identities, exact hashes and retained near-duplicate fingerprints. Six new SDXL and six new Kandinsky photo groups passed the frozen source qualification, including all 48 publisher assets. Each source supplies two edit types and six controls in both formats. PNG and JPEG75 use the same precompression pixel truth. The study retains ordinary processing as a control even when it changes many pixels. [Prior matched panel and source rights](image-background-patch-transfer-2026-11-11.md) · [Blend construction](image-boundary-blend-transfer-2026-11-11.md)

| Panel | Format | Check | Sharp /12 | Feathered /12 | Controls highlighted /72 | Originals highlighted /12 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Earlier | PNG | Simple | 0 | 1 | 1 | 0 |
| Earlier | PNG | Fitted | 0 | 0 | 24 | 4 |
| Earlier | JPEG75 | Simple | 1 | 1 | 2 | 0 |
| Earlier | JPEG75 | Fitted | 0 | 0 | 43 | 7 |
| New sources | PNG | Simple | 0 | 0 | 8 | 1 |
| New sources | PNG | Fitted | 0 | 0 | 18 | 3 |
| New sources | JPEG75 | Simple | 0 | 0 | 2 | 0 |
| New sources | JPEG75 | Fitted | 0 | 0 | 12 | 2 |

A useful location covers at least half the known edited pixels and reaches 0.30 intersection-over-union. The prior current routes found zero feathered edits in the earlier panel and produced 11 PNG and seven JPEG75 control highlights. The prospective holdout tested the two candidates at their sealed points. It made no additional current-route model calls. An independent local readback reconstructed all 384 encoded inputs, checked all 192 fresh truth-mask commitments and recomputed 768 region outcomes from native pixels. [Candidate runner](../backend/scripts/image_wavelet_region_transfer.py) · [Fresh qualification and replay](../backend/scripts/image_wavelet_region_holdout.py)

From `backend`, run `python -m scripts.image_wavelet_region_transfer --help` or `python -m scripts.image_wavelet_region_holdout --help` for the staged commands. The aggregate records the exact six feature weights, four points, source roster commitments and execution environment. Save its `points` value with sorted keys, two-space indentation and one terminal newline to reproduce the sealed points file. Exact replay requires the locally retained source rosters or an equivalent rights-qualified reconstruction. The source media stay in the development archive.
