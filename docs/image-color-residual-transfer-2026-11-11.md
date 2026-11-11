# Local color-residual correlations on blended image edits

Two color-residual checks found three useful edit locations across the earlier
panel and none across the later panel. They also highlighted more ordinary
views than the earlier reference. The checks remain experimental. [Counts and
sealed parameters](evaluation-panels/image-color-residual-transfer-2026-11-11.json)

The test measures whether local color dependence changes where a generated
patch replaces part of a photograph. It uses no external code or weights.
The recent color-forensics papers motivate comparing channels instead of
measuring just the magnitude of fine detail. Their image-level methods and
results do not establish localization of these edits. [CHROMA](https://arxiv.org/html/2606.08864v1) · [Noise-residual artifact study](https://arxiv.org/html/2608.20558v1)

The runner uses two Gaussian residual bands in RGB and Lab, measured on
16-pixel native cells. The features are three regularized correlations per
band and two horizontal-to-vertical lag contrasts per color space. Each
candidate compares the interior of a native rectangle with a separated outer
ring. One rule ranks the standardized magnitude of this difference. A second
ranks a fitted direction using paired edited and ordinary versions from eight
photos. Eight other photos set a per-rule point for each delivery format using
the second-largest ordinary source maximum. The code, source receipts and
operating procedure were sealed before fitting. Both points were sealed before
scoring the two comparison panels.

The panels use 12 disjoint photos each. Their sources were outside fitting
and point selection for this method. They had prior project evaluation
exposure. Sharp and feathered inserts use the exact original background
outside the edit. Six controls per photo include unchanged, ordinary blur and
contrast, and local or whole-image reconstruction. A useful result must cover
at least half the changed pixels and have at least 0.30 intersection over
union. The earlier reference reuses observations bound to the same PNG and
JPEG75 bytes. [Photo attribution and terms](image-background-patch-transfer-2026-11-11.md) · [Source separation and exact delivery](image-wavelet-region-transfer-2026-11-11.md)

| Panel | Delivery | Check | Sharp /12 | Feathered /12 | Controls flagged /72 |
| --- | --- | --- | ---: | ---: | ---: |
| Earlier | PNG | Local deviation | 1 | 1 | 36 |
| Earlier | PNG | Fitted color | 0 | 0 | 37 |
| Earlier | PNG | Earlier reference | 0 | 0 | 11 |
| Earlier | JPEG75 | Local deviation | 0 | 0 | 27 |
| Earlier | JPEG75 | Fitted color | 0 | 1 | 47 |
| Earlier | JPEG75 | Earlier reference | 1 | 0 | 7 |
| Later | PNG | Local deviation | 0 | 0 | 17 |
| Later | PNG | Fitted color | 0 | 0 | 20 |
| Later | JPEG75 | Local deviation | 0 | 0 | 24 |
| Later | JPEG75 | Fitted color | 0 | 0 | 36 |

A separate readback reconstructed all 384 exact deliveries and masks,
recomputed intersection counts from recorded rectangles and source masks,
and matched the archived reference. With the review points ignored, the
highest-ranked rectangle on the later panel could localize only two of 48
edits for the fitted rule and zero for the magnitude rule. The ranking itself
limits how many edits a lower point could recover. The comparison uses locally retained
rights-qualified files. Run `python -m scripts.image_color_residual_transfer
--help` from the backend directory for the three-stage recipe.
