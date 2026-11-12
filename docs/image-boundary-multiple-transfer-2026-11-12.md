# Two separated sharp regions in one PNG

The existing boundary check found one useful location in each of 12 PNGs with two known sharp pastes. The new search can locate both. At the existing first-region point it found both in 11 of the 12 edited photos and added a second region in all 36 ordinary Chromium captures. A fixed-color frame check reduced the additional screenshot highlights to 13, still spanning all 12 interface groups. After choosing secondary points on separate sources, neither candidate found both edits in any held image. Both remain experimental. [Audited aggregate and frozen panel](evaluation-panels/image-boundary-multiple-transfer-2026-11-12.json)

## Regions and controls

The candidate keeps the existing top-scoring rectangle and only considers another when that first region passes its current review point. It searches the same nine rectangle dimensions, with at least 20 native pixels of clearance between the boxes in one axis. Each side compares native RGB steps with parallel steps three pixels away. One arm uses that four-sided score directly. The second suppresses the additional region if the same exact RGB color covers at least four-fifths of a rim sample on three sides. It samples each rim within one pixel of the boundary and omits the three pixels at each corner. A useful region reaches 0.30 intersection over union and 0.50 recall against its exact changed pixels. A useful pair requires separate matches to both edits. [Existing native boundary check](image-boundary-prominence-holdout-2026-11-11.md) · [Candidate runner](../backend/scripts/image_boundary_multiple_transfer.py)

Thirty-six rights-reviewed Wikimedia Commons photos fill disjoint roles. Eight photo groups and four Chromium layout groups set each secondary point from their second-highest control-source maximum, with the existing point as the floor. The 12 held photo groups supply two sharp donor pastes, a sharp paste paired with a feathered paste, the two separate single-paste views, and four ordinary views per group. Host and donor photos were selected without scores and reused only within their assigned fit or held cohort. A third source stratum supplies backgrounds for the Chromium layouts. The 12 held layouts each have three variants with unmodified interface panels. They include text overlays, bordered tiles, and borderless photos. These controls intentionally align rectangular dimensions with the search grid. The layout families and captured files were bound before any fit or held scoring. Source photographs had earlier project evaluation exposure. The controlled paste layouts and screenshots are new, and the screenshot sample does not estimate the frequency of interface designs in customer uploads. [Source attribution and terms](image-affine-native-scale-2026-11-11.md) · [Chromium capture runner](../backend/scripts/capture_image_boundary_multiple.cjs)

## Held comparison

| Secondary check | Useful pairs, two sharp /12 | Useful pairs, sharp and feathered /12 | Extra ordinary photo regions /48 | Extra screenshot regions /36 | Screenshot groups with extra regions /12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Separated rectangle, fit point | 0 | 0 | 0 | 0 | 0 |
| Frame excluded, fit point | 0 | 0 | 0 | 1 | 1 |
| Separated rectangle, first-region point diagnostic | 11 | 0 | 6 | 36 | 12 |
| Frame excluded, first-region point diagnostic | 11 | 0 | 6 | 13 | 12 |

The prespecified release criteria require at least six complete sharp pairs, at most two additional ordinary photo regions from at most two groups, at most two additional screenshot regions from at most one group, and at most one single-paste group with an incorrect additional location. The separate points are 3.0 without the frame rule and approximately 2.612 with it. The last two rows are a post-evaluation diagnostic at the existing first-region point of approximately 0.829. Each added ordinary photo region is counted even when the first result already highlighted that file. At the lower point, four of the 24 single-paste views also gained an incorrect location. The rim check removes all 12 bordered tile highlights and 11 of 12 overlay highlights while preserving the 11 complete sharp pairs. It removes zero of 12 borderless layout highlights, where ordinary photo content creates the same local discontinuities. The second-highest borderless screenshot score was 2.528, above the strongest sharp-pair second score of 2.175. Every point that limits borderless screenshot highlights to one group also removes all sharp pairs on this panel. No useful feathered location emerged at the lower point. These findings measure native seam geometry in conventionally composed PNGs. They provide no AI authorship probability.

## Local verification

The native replay rebuilt all 192 PNG and truth commitments, independently reconstructed 323 proposal crops with Pillow, and checked the prefix-sum response by direct edge samples. The portable checker recalculates fit points, useful overlaps, group counts and diagnostic outcomes using the public ledger without access to any source media. The initial freeze is `ec49831671da0cfac8377562fc37c604e4ea1d5ed824fdbb45ea1f8dca514e62`. The portable verifier was corrected to allow native float32 rounding above the mathematical cap after two rendered tile scores reached 3.00000024. The final native replay uses the corrected verifier fingerprint and reproduces the original fit, points, held measurements and outcome exactly. [Portable checker](../backend/scripts/review_image_boundary_multiple_transfer.py)

```sh
python3 -I -S backend/scripts/review_image_boundary_multiple_transfer.py \
  docs/evaluation-panels/image-boundary-multiple-transfer-2026-11-12.json
```

For a native run, use the retained source archive with `plan.json`, `acquisition.json` and 38 source photographs, and a new local output folder. Chromium needs the frontend Playwright dependency and local Chromium installation. The capture renderer blocks browser network requests.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_boundary_multiple_transfer prepare "$archive" "$run"
node backend/scripts/capture_image_boundary_multiple.cjs plan "$run"
node backend/scripts/capture_image_boundary_multiple.cjs capture "$run"
for stage in freeze fit held verify; do
  backend/.venv/bin/python -B -m scripts.image_boundary_multiple_transfer "$stage" "$archive" "$run" || exit 1
done
```
