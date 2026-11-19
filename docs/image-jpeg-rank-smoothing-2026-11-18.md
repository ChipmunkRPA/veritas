# Rank-order smoothing around small edits in a single JPEG

Two checks based on short, nearly flat runs located none of 160 staged
edits, even before applying their score cutoffs. Every edit had at least
one rectangle that met the overlap rule and the checks’ own local support
requirements. The checks consistently ranked other areas higher. They
remain outside the app.

## Frozen measurement

The experiment reads native luminance from one upright JPEG. For each
horizontal and vertical five-pixel sequence that stays within an
eight-pixel JPEG block, it compares the middle three values with the two
outer steps. A nearly flat run has at most two combined levels of change
across the three middle values and outer steps of at least two levels
in the same direction. A second reading counts a strict sign reversal
between successive middle differences. The calculation excludes near-clipped
values. It compares these frequencies within four matched contrast ranges
defined by the absolute difference between the two endpoints.

Candidate rectangles use the native 16-pixel grid, side lengths from 32
to 192 pixels, an aspect ratio of at most two and a surrounding 32-pixel
ring. The score uses the smaller excess in nearly flat runs across the
two directions. The guarded reading also requires fewer sign reversals
in both directions. Both directions need shared contrast support between
the rectangle and its ring. The contrast bin edges are 5, 11, 23,
49 and 97 luminance levels. Each included bin requires at least 16
samples inside and 20 in the ring, and uses the smaller count as
its weight. Each direction needs at least 120 weighted samples,
at least 160 ring samples, and support for at least 55% of its
interior samples. The guarded score caps the first score at 0.55
times the smaller decrease in the reversal rate. The new readings
use no learned weights.
Primary research finds that simple median-filter streaks can disappear
after JPEG encoding and that different smoothing operations can produce
similar residuals. Recent JPEG inpainting work identifies related losses
of high-frequency evidence. [Median-filter study](https://ws2.binghamton.edu/fridrich/Research/SPIE_median.pdf)
· [JPEG inpainting study](https://doi.org/10.1109/TIP.2026.3699086)

Twelve photos supply 240 ordinary calibration deliveries. Each reading
and quality uses the second-highest ordinary source maximum among cases
the existing checks left clear. The highest rectangle must exceed that
point. The held banks use 12 and eight separate photos with no shared
recorded photographer, source file or decoded-image identity between
calibration and the two banks. The photographs had earlier project
exposure. Each held group supplies sharp and feathered LaMa and MI-GAN
removals, plus matched unchanged, local blur, tone and copying controls.
The 12-photo bank also includes conventional fills. Both JPEG qualities
share the same held photos and edits. The test uses pre-encoding masks,
at least 0.30 intersection over union and at least half the changed
pixels for a useful location. [Photo qualification and editor terms](image-jpeg-frequency-bands-2026-11-17.md#held-photographs-and-test-controls)
· [Earlier camera-panel intake](image-jpeg-pscc-localization-2026-11-17.md)

## Held result

| JPEG quality | Reading | Top useful before cutoff | Added useful locations | Added ordinary highlights | Wrong edit highlights with no useful incumbent region | Top boxes overlapping the unchanged top box by at least 0.50 IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 75 | Nearly flat runs | 0/80 | 0 | 3/184 | 4 | 64/80 |
| 75 | Runs and fewer reversals | 0/80 | 0 | 0/184 | 0 | 66/80 |
| 90 | Nearly flat runs | 0/80 | 0 | 42/184 | 23 | 71/80 |
| 90 | Runs and fewer reversals | 0/80 | 0 | 16/184 | 12 | 74/80 |

The same exact inputs gave the existing aligned JPEG and regional
checks three useful JPEG75 regions and zero at JPEG90. Those checks
highlighted 29 of 184 ordinary JPEG75 and 15 of 184 ordinary JPEG90
deliveries. The 42 additional JPEG90 highlights from the unguarded
reading covered five photos. The 16 from the guarded reading covered
two. The 184 controls per quality include repeated processing variants of
the same 20 photos. These counts measure the two compared app checks,
including their JPEG90 repeat view.

A native recount re-opened 768 files and their exact masks, then
enumerated the permitted rectangles for all 160 edited deliveries.
Every edit still had qualifying geometry when the matched-contrast
support rules were applied to its actual JPEG. The cutoffs determine
whether to show a highest-ranked rectangle. They cannot move that
rectangle into the edited area. The selected boxes often stayed on
the same scene features after the edit. An authored local median filter
and a matched Gaussian filter also failed to move the top rectangle
into the intended region at either quality.

A distinct follow-up could plant the same operation at two different
texture-matched locations in each photo and test whether the top region
moves with the operation. Randomizing the locations before measurement
would distinguish a cue that responds to the intervention from one that
repeatedly selects a distinctive part of the same scene.

## Recount

The [grouped receipt](evaluation-panels/image-jpeg-rank-smoothing-2026-11-18.json.gz)
contains sorted calibration group maxima, anonymous held group counts,
and aggregate native commitments. Images, masks, source identifiers,
and coordinates stay local. The standard-library reader recomputes the
cutoffs, coverage, reported counts and frozen advancement decision.

```sh
python3 -I -S backend/scripts/review_image_jpeg_rank_smoothing.py \
  docs/evaluation-panels/image-jpeg-rank-smoothing-2026-11-18.json.gz
```
