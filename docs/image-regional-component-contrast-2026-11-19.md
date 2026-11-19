# Local context for highlighted image regions

Checking the response around an existing region removed seven ordinary
highlights from the native PNG panel and improved one useful edit location.
On a separate PNG panel, the same rule removed ten ordinary highlights
and lost one of six useful locations when added to the local app's current
horizontal-repeat check. The rule stays outside the app.

The check compares the median response inside each retained connected
region with the 90th percentile over reliable pixels in an exterior ring.
The ring stretches from two to eight map pixels from the selected region.
The rule suppresses a region when its response rises by less than 0.10
above that comparison. It retains regions with fewer than 32 reliable ring
cells or with less than half the clipped ring measured. Selected pixels
otherwise retain their original geometry. No model training or fitting
uses these responses. Recent work also identified weak or naturally smooth
surroundings as a localization failure mode, and used matched processing
controls to isolate synthetic manipulation from ordinary edits.
[Local anomaly research](https://arxiv.org/abs/2606.02178)
· [Processing-matched benchmark](https://arxiv.org/abs/2607.26232)

| Panel | Check | Useful locations | Misplaced highlights on edited images | Ordinary highlights |
| --- | --- | ---: | ---: | ---: |
| Native PNG | Existing single pass | 45/59 | 8/59 | 34/177 |
| Native PNG | Add local contrast | 46/59 | 5/59 | 27/177 |
| Other PNG sources | Existing single pass | 4/52 | 36/52 | 79/104 |
| Other PNG sources | Add local contrast | 4/52 | 29/52 | 59/104 |
| Other PNG sources | Existing horizontal repeat | 6/52 | 26/52 | 50/104 |
| Other PNG sources | Horizontal repeat plus local contrast | 5/52 | 18/52 | 40/104 |

The native comparison reproduced all 236 inputs and archived responses.
It kept all 45 useful highlights and removed an unrelated component to
make one additional region useful. A separate, fixed lower-response
addition produced no differences. The second panel used 26 distinct
BG-REAL photos with three editor families. It reproduced all 156 original
inputs and measured their regional response again. The comparison reused the
prior exact horizontal-repeat memberships after verifying the original
component pixel counts and truth overlaps. The source groups were separate
from the 236-image comparison, with known prior project exposure. Both panels
require at least half the known changed pixels and 30% intersection over
union. The follow-up lost one useful removal region from the horizontal
repeat and localized no additional edits. These results describe the
location of changes and supply no AI-authorship calibration.
[Native source panel](image-edge-context-transfer-2026-11-14.md)
· [Separate PNG panel and source licenses](image-edge-context-new-sources-2026-11-14.md)

The public receipt contains selected and overlapping pixel counts and
commitments to the tested files and prior panels. The independent
standard-library recount verifies all 392 input keys and recalculates
the control alerts and localization results. It uses no image pixels or
weights. [Numeric receipt](evaluation-panels/image-regional-component-contrast-2026-11-19.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_regional_component_contrast \
  docs/evaluation-panels/image-regional-component-contrast-2026-11-19.json
```

## Extending the measured boundaries

A separately frozen comparison began with the existing selected regions.
It extended them through immediately adjacent reliable pixels above a
lower response point, for a maximum of eight steps and 25% more selected
area per component. On the second panel, a separate arm required each
new pixel to qualify in the aligned horizontal repeat too. The existing
pixels, model weights, and eligibility stayed fixed.

| Panel | Check | Useful locations | Ordinary alerts | Ordinary area highlighted in pixels |
| --- | --- | ---: | ---: | ---: |
| Native PNG | Existing single pass | 45/59 | 34/177 | 915,779 |
| Native PNG | Extend boundaries | 45/59 | 34/177 | 1,048,687 |
| Other PNG sources | Existing horizontal repeat | 6/52 | 50/104 | 2,522,708 |
| Other PNG sources | Extend from the original view | 6/52 | 50/104 | 2,847,162 |
| Other PNG sources | Extend where both views support it | 6/52 | 50/104 | 2,694,557 |

The replay matched all 236 archived native responses and reran the original
and applicable reflected responses on the 156 separate PNGs. It reproduced
the exact previous component overlap for every reflected view. None of the
new masks changed which locations met the edit-overlap criterion. Ordinary
marked areas grew, including in the two-view version. Boundary extension
stays experimental. [Boundary growth receipt](evaluation-panels/image-regional-boundary-growth-2026-11-19.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_regional_component_contrast \
  docs/evaluation-panels/image-regional-boundary-growth-2026-11-19.json
```
