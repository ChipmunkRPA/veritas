# Single-file image localization candidates

Two additional approaches were tested offline. Neither changed the app's general-purpose image checks. [Evaluation receipt](evaluation-panels/image-local-edit-candidates-2026-11-12.json)

## Comparing texture spectra

The candidate measures how signed high-frequency pixels in overlapping 4 × 4 neighborhoods vary together. It compares the eigenvalue distribution of each cell's trace-normalized covariance matrix with similar-looking, distant cells. Luma, opponent-color channels, and agreement between both were tested separately. Each arm used an operating point fixed on 60 ordinary views. The idea draws on patch self-similarity research. The implementation uses basic numerical operations and no external model weights. [Patch self-similarity research](https://www.sciencedirect.com/science/article/pii/S0262885624002427) · [Recent inpainting evaluation](https://arxiv.org/html/2602.00192v2)

The held panel contained 12 LaMa and six FLUX source groups with matched original, sharp edit, feathered edit, local blur and local tone deliveries in PNG and JPEG75. All three arms gave zero useful locations for the 36 sharp and 36 feathered deliveries. Before applying the operating points, the best color or agreement rectangles aligned with two sharp PNG edits and one feathered PNG edit. None aligned on JPEG75. Luma and agreement each highlighted an additional ordinary source group. Color highlighted seven. The retained source photographs had earlier project exposure and the tested measurements were new. [BG-REAL source rights](image-bg-real-roster-v218-method.md) · [Retained FLUX compositions](image-flux-photo-expansion-v285-result.md)

## Between-block copies in JPEG95

The existing copy cascade was tested on the exact standard quality-95 quantization tables. The first panel used a 56-pixel minimum native crop side and a second source-separated panel used 32 pixels. The source selection excluded the recorded edit contributors and source hashes from the earlier JPEG75, JPEG85 and JPEG90 tests. The architecture and repeated-structure controls reuse commercially permitted Wikimedia sources from the project archive. Before encoding, exact target and donor boxes and changed-pixel masks were committed for each edit. Both panels used matched untouched, local tone and local blur controls. [Existing JPEG85 check](image-jpeg-offgrid-quality85-2026-11-12.md) · [Public source roster](image-affine-native-scale-2026-11-11.md)

| Crop minimum | Sharp useful, current → candidate | Feathered useful, current → candidate | Added flags on ordinary controls |
| --- | ---: | ---: | ---: |
| 56 pixels | 13/24 → 14/24 | 4/12 → 5/12 | 0/48 |
| 32 pixels | 6/24 → 8/24 | 2/12 → 3/12 | 0/46 |

The two studies replayed exact native crops, including tests for altered coordinates and measurements. Both panels fell below the fixed requirement of four added sharp source groups. The tested between-block method measures copied content, with no indication of which area came first and zero weight in the AI authorship assessment.
