# Photo admission for small image-boundary checks

The small-boundary fallback localized 13 additional conventional pastes on 12 new target photos. A flat-block rule preserved all 13 locations but removed only two of nine added screenshot flags. All seven added flags on photo-rich screenshots survived, across six source groups. The frozen test keeps the fallback experimental. [Audited roster and case ledger](evaluation-panels/image-small-boundary-screenshot-admission-2026-11-12.json)

| Held outcome | Current check | With fallback | With fallback and photo admission |
| --- | ---: | ---: | ---: |
| Same-photo paste, correct location | 2/12 | 7/12 | 7/12 |
| Separate-donor paste, correct location | 1/12 | 9/12 | 9/12 |
| Ordinary photo views with any box | 3/48 | 3/48 | 3/48 |
| Screenshots in the sealed stratum with a box | 0/50 | 9/50 | 7/50 |
| Distinct screenshot groups in that stratum with a box | 0/34 | 7/34 | 6/34 |

The 13 extra locations came from ten target photos. Both experimental arms added zero incorrectly located paste boxes. The existing 137 screenshot flags in the full deduplicated roster remained in all arms. The gate applies only when the current check produces no box.

## Frozen admission and source roles

Keep every fixed-size result. Otherwise apply the published 56-pixel minimum, 18% maximum on each dimension, and strict distributed-side score of 1.2805722063718288. The gate skips this fallback when at least 25% of the original 8 × 8 RGB blocks are constant in every channel. The 25% value comes from the existing content-kind code and was frozen before candidate scoring. The test required at least six useful new boxes across five photos, at least two in each paste type, at most one lost useful box, at least 75% fewer added screenshot flags and at most two residual screenshot groups. A useful location reaches 0.30 intersection over union and covers at least half of the exact changed pixels. [Published fallback](image-boundary-extent-transfer-2026-11-12.md) · [Pre-existing flat-block measurement](../backend/app/engine/fusion/content_kinds.py)

The screenshot roster uses exact Chromium viewports of 704 × 544 and 768 × 512. The published examples come from Bootstrap, the GOV.UK Design System and the U.S. Web Design System. Local HTML fixtures use separate credited photos as the background for four layout types resembling articles, galleries, listings and photo editors. These combine a detailed image with ordinary rectangular interface elements. No raw capture or source photograph is redistributed. Bootstrap permits commercial reuse of its MIT code and CC BY documentation. GOV.UK provides MIT example code and Open Government Licence documentation. The USWDS uses public-domain and CC0 terms for its own work and lists third-party exceptions. [Bootstrap terms](https://getbootstrap.com/docs/5.3/about/license/) · [GOV.UK code and site terms](https://design-system.service.gov.uk/components/) · [USWDS terms](https://designsystem.digital.gov/about/contribute/#licenses-and-attribution)

The 12 target, 12 donor and 12 composite photos came from distinct credited photographers and recorded cache components. All selected photos carry CC BY-SA 2.0 terms. Known prior roles in the boundary and other image trials were excluded. A broad unrelated geometric-copy proposal test may have examined these source photos. Its scores played no role in this selection. The photo controls comprise unchanged source views, global tone changes, feathered local blur and resize round trips. Camera provenance and unseen near-duplicate relationships remain unverified. [Photo source inventory](image-fresh-source-cache-v39-results.md)

## Incumbent-only seal

The first intake captured 118 of 120 planned views. Fifteen exact RGB duplicates shared the GOV.UK navigation view. The remaining 103 views gave 21 qualifying negative views across 16 source groups in two families. The predeclared minimum was 20 groups across three families. No candidate scores were read or produced. A new reserve froze distinct Bootstrap layouts and native viewport captures focused on government example bodies. Focused and full-page captures of the same component share a source group. The reserve kept the original source pixels, scores, boxes and gate criteria.

The combined intake captured 202 of 212 planned views and retained 187 unique RGB views from 72 captured source groups. Five planned layouts failed to render sufficient text or their focused example selector. The incumbent-only run sealed all 50 views with no existing box before scoring the flexible candidate. This intentionally measures cases that can reach the fallback. Its flag rates describe this stratum of the selected controls.

| Screenshot family | Distinct views | Incumbent flags | Views in sealed stratum | Groups in sealed stratum | Added fallback flags | Added flags with gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bootstrap | 56 | 32 | 24 | 16 | 2 | 0 |
| GOV.UK | 47 | 30 | 17 | 10 | 0 | 0 |
| USWDS | 60 | 60 | 0 | 0 | 0 | 0 |
| Local photo-rich layouts | 24 | 15 | 9 | 8 | 7 | 7 |

The photo-rich screenshots show where whole-image flatness fails as an admission cue. A detailed background can lower the flat-block share while ordinary interface panels supply sharp local boundaries. The test measures conventional boundary localization and carries zero AI confidence weight.

## Verify and reproduce

The independent reader verified all 202 original capture hashes, the preserved initial incumbent seal, all 122 scored PNG deliveries, 122 independently recomputed flat-block measurements and 36 arm-specific native crop records. The public ledger can be read with the Python standard library and without any weights or media.

```sh
python3 -B -I -S backend/scripts/review_image_small_boundary_screenshot_admission.py \
  docs/evaluation-panels/image-small-boundary-screenshot-admission-2026-11-12.json
```

For native replay, use the combined frozen `screenshot_plan`, `screenshot_captures` and `reserve_receipt` as `screenshot-plan.json`, `screenshot-captures.json` and `reserve-receipt.json`, respectively. Write sorted keys, two-space indentation and a final newline. Use the frozen `roster` object for `$roster`. Supply the exact capture PNGs under `$run/captures/` and each attributed source JPEG as `$cache/<encoded_sha256>.bin`. The [capture runner](../backend/scripts/capture_image_small_boundary_screenshots.cjs) records a new intake from the listed public pages. Live pages can change, and this new intake requires its own seal.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze incumbent candidate; do
  backend/.venv/bin/python -B -m scripts.image_small_boundary_screenshot_admission "$stage" "$cache" "$roster" "$run" || exit
done
backend/.venv/bin/python -B backend/scripts/review_image_small_boundary_screenshot_admission.py "$run"
```
