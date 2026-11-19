# Learning ordinary JPEG changes from paired camera images

Neither candidate localized any of the 152 held AI-removal deliveries,
even when the highest-ranked rectangle was counted regardless of score.
Suppressing variation learned from ordinary processing reduced false alerts
in three of four comparisons. It added no correctly localized edits over
the current checks, so this method stays experimental.

## Method

Each 16-pixel cell contributes 18 log frequency energies across luma and
two chroma channels. Its residual is the difference from the median of
nearby cells in a surrounding ring. Highly clipped cells and cells without
a complete ring stay outside the measurement. Both arms use the same
features, window shapes and requirement for three quarters of the cells in
a window to be measurable. One arm normalizes the variation in unretouched
camera photos. The other also suppresses directions associated with common
processing. It learns those directions from aligned originals and manually
blurred, color-adjusted or copied counterparts, with equal weight per
source and treatment. Only the highest-magnitude fifth of supported paired
changes enters that covariance estimate. All development fitting uses
ordinary camera examples. This tests a statistical residual method inspired
by real-only forensic training. [Forensic Twins preprint](https://arxiv.org/abs/2609.31514)

Twelve camera sources supplied the fit, and another twelve supplied separate
operating points for JPEG75 and JPEG90. The point for each arm is the second
highest ordinary per-source maximum among files left clear by the current
checks. The held inputs cover 12 further distinct camera photos and 14
separate scene-text groups, all previously used in other project experiments.
The scene-text references can have unknown provenance. No held examples
or generated examples entered the fit or operating points. The [source and
commercial-use notes](image-masked-reconstruction-2026-11-17.md#held-inputs)
cover the retained input banks.

| Held panel | JPEG quality | Current useful over edited | Added useful, either arm | Added ordinary alerts, normalized / suppressed |
| --- | ---: | ---: | ---: | ---: |
| Camera photos and matched controls | 75 | 3/48 | 0 | 40/120 · 40/120 |
| Camera photos and matched controls | 90 | 0/48 | 0 | 30/120 · 20/120 |
| Scene-text removals and controls | 75 | 0/28 | 0 | 7/70 · 6/70 |
| Scene-text removals and controls | 90 | 0/28 | 0 | 8/70 · 5/70 |

A useful location covers at least half the changed pixels and reaches 0.30
intersection over union. Multiple deliveries share one source. For example,
the 40 extra JPEG75 camera alerts for either arm came from four photos.
The held result measures this particular small-edit and JPEG test.

A separate replay checked the original and mask hashes for all 532 held
deliveries and recomputed 1,064 crop intersections. The public receipt
preserves anonymous source grouping, mask area, crop overlap and alerts.
[Numeric receipt](evaluation-panels/image-paired-camera-residuals-2026-11-19.json)

```sh
python3 -B -I -S backend/scripts/review_image_paired_camera_residuals.py \
  docs/evaluation-panels/image-paired-camera-residuals-2026-11-19.json
```
