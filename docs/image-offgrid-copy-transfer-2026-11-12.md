# Localizing copies between JPEG block boundaries

The current image comparison can miss small copied areas when their displacement crosses the JPEG encoding grid. This experiment uses a uniform grid of overlapping image patches to propose integer displacements. An aligned rectangle needs agreement in both coarse brightness and finer texture. After smoothing each RGB channel, the candidate also requires a mean inside-crop color difference of at most 3.2 levels on a 0–255 scale, a 90th percentile of at most 6 levels, and stronger disagreement on at least three of four outer strips. This measures a distinct change from matching pixels inside a crop to different surroundings. Published work explores dense correspondences and neighborhood behavior around matched regions. The implementation uses project code and no external weights. [Dense correspondence study](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2014/Papers/1569902465.pdf) · [Matched boundary study](https://www.gipsa-lab.grenoble-inp.fr/~kai.wang/papers/ASOC21_preprint.pdf)

## Frozen local comparison

The twenty edited photo groups exclude source hashes, linked groups and recorded authors used in the earlier same-orientation and encoded-brightness comparisons. The selected archive photos have recorded CC BY 2.0 or CC BY-SA 2.0 terms. Each source supplies 40 × 48 and 64 × 64 sharp copies at predetermined off-grid displacements, including short and long translations. Both crops need at least 0.50 intersection-over-union, and the target crop must cover at least half the authored changed pixels. The 240 views include 80 authored edits and 160 ordinary controls. The ordinary set contains native images, re-encodes, contrast and local blur, and JPEG versions of ten separate source photos from the architecture and building-exterior rosters. Their rights and exact acquired bytes were already documented. [Archive qualification](image-fresh-source-cache-v39-results.md) · [Earlier JPEG comparison](image-jpeg-copy-blocks-transfer-2026-11-12.md) · [Architecture controls and rights](image-affine-native-scale-2026-11-11.md)

The comparison runs the existing translation and exact JPEG brightness-block checks first. It accepts a new pair only when that path reports a completed comparison with no pair. The gate requires additional useful locations on at least four source groups at each size. It allows up to one additional ordinary photo group, zero additional architecture groups and up to two incorrectly localized edited views. The measurements concern deliberately authored copy-and-paste. They carry zero weight in AI authorship confidence.

| Delivery | Authored crop | Current useful | Current plus candidate | With a 32 px minimum, exploratory |
| --- | --- | ---: | ---: | ---: |
| JPEG75 | 40 × 48 | 0/20 | 8/20 | 6/20 |
| JPEG90 | 40 × 48 | 1/20 | 10/20 | 8/20 |
| JPEG75 | 64 × 64 | 3/20 | 13/20 | 13/20 |
| JPEG90 | 64 × 64 | 8/20 | 14/20 | 14/20 |

The frozen candidate added 33 useful edited views across ten photo groups at each size, with zero new incorrect edit locations. The existing comparison flagged four ordinary views. The combined method flagged eight. The four added flags came from three versions of one photo and one version of an architecture photo. The architecture flag exceeded the frozen gate. A subsequent hash audit found two source photos in other older project experiments. Excluding both leaves nine source groups at each size with useful gains.

All four new ordinary flags had crop short sides of 28 or 30 px. Requiring each displayed crop to measure at least 32 px on both axes retained 29 useful edited views and removed all four additional ordinary flags in this panel. This calculation used the original scores and informed a separate source-disjoint confirmation. The original 240 result records replayed byte for byte, including their reported crop coordinates and measurements. The current source has an import-only formatting cleanup between the two passes. Both freezes contain the same truth and source records.

## Separate confirmation

Freeze the 32 px minimum before scoring any of the confirmation sources. The sixteen photo groups exclude the earlier comparisons, all twenty first-trial groups, and both audited older exposure rosters. The fourteen separate repeated-structure controls come from the licensed abandoned-building and cityscape roster. Wide images are resized and center-cropped without changing their aspect ratio. Keep the same translation search, corroboration and acceptance gate.

| Delivery | Authored crop | Current useful | Current plus candidate |
| --- | --- | ---: | ---: |
| JPEG75 | 40 × 48 | 0/16 | 5/16 |
| JPEG90 | 40 × 48 | 0/16 | 6/16 |
| JPEG75 | 64 × 64 | 0/16 | 5/16 |
| JPEG90 | 64 × 64 | 3/16 | 6/16 |

The confirmation passed. It added 19 useful views on six small-copy source groups and five medium-copy groups. It added zero incorrect edit locations or flags on 140 ordinary deliveries, including 28 versions of 14 repeated-structure sources. The earlier panel's additional ordinary matches show why the comparison remains evidence of local similarity. The controlled edits were sharp translations with no separate scaling, rotation or feathering.

## Local app and verification

Enable the fallback with `VERITAS_IMAGE_JPEG_OFFGRID_COPY_REVIEW=true` in the development environment. It runs when the preceding translation and encoded-block checks report no pair. It accepts upright color 4:2:0 JPEGs with a short side of at least 384 px, a long side of 513–1,024 px, and the two standard compression tables used for JPEG75 and JPEG90 in this test. The search can stop when excessive competing offsets prevent a distinct comparison. If the best qualifying pair is smaller than 32 px on either axis, the check returns no pair. The app and PDF show the exact native crop coordinates, their colors and the measured differences to their surroundings. The JSON retains full measurement precision. Either region may have come first.

The exact serving code evaluated eligibility on all 204 confirmation inputs and reproduced those records byte for byte on replay. It returned no pair on all 124 assessed ordinary deliveries. Sixteen ordinary deliveries fell outside the serving profile, including five previously assessed native controls. The added offset budget preserved all qualifying pairs and measurements. The original experiments and serving replay have distinct committed hashes. [Frozen results and serving receipt](evaluation-panels/image-offgrid-copy-transfer-2026-11-12.json)

The normal Full analysis processed a held positive and a matched unchanged JPEG with simulated prepaid checkout. The positive returned two source-bound 32 × 42 crops. Their mean inside difference was 1.24537 on the 0–255 RGB scale. The four surrounding strips had means of 14.24117, 40.29295, 41.18431 and 24.46819. The unchanged JPEG returned a completed check with no pair. Crop coordinates, measured values and decoded crop pixels agreed across the API, JSON, PDF and signed evidence package. Original-file replay rejected a different source, a moved crop, and an altered color measurement. Both signed packages also verified with network and application runtime imports disabled.

## Reproduce

Use the existing source cache and the archived licensed photo ledgers. Set `cache` to the v39 private groups JSON, `blocks` to the previous JPEG block freeze, `translation` to the earlier translation freeze, `transfer` to the v298 private freeze, `expansion` to the v285 private inputs, and `controls` to the Wikimedia architecture intake folder containing the bound plan, acquisition receipt and source JPEGs. Set `first_run` and `confirmation_run` to two unused scratch paths. The photo bytes remain in the evaluator's workspace. The confirmation validates the exact original first-trial source, truth and mask commitments, allowing it to use an equivalent first-trial freeze generated with the current runner.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_offgrid_patch_transfer freeze \
  "$first_run" "$cache" "$blocks" "$translation" "$controls"
for stage in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_offgrid_patch_confirmation \
    "$stage" "$confirmation_run" "$cache" "$blocks" "$translation" "$first_run/freeze.json" \
    "$transfer" "$expansion" "$controls" || exit
done
```
