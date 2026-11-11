# Sharp image boundaries on a second source panel

The earlier native boundary test localized 5/12 sharp PNG insertions, with 13/72 ordinary control highlights. The existing regional route found 0/12 locations and highlighted 11/72 controls on those same PNGs. Both missed every feathered insertion. This follow-up checks the fixed model-free PNG operating point on a second panel and directly tests what conventional hard pasted boundaries can show. [First comparison](image-boundary-blend-transfer-2026-11-11.md) · [Audited result](evaluation-panels/image-boundary-prominence-holdout-2026-11-11.json)

## Frozen PNG replay

Measure native RGB steps along the four sides of each candidate rectangle and compare them with parallel steps displaced three pixels. Use the original rectangle search, code and PNG operating point with no new fitting. The operating point is the second-largest ordinary control source maximum from eight earlier development groups. To pass this comparison the candidate needs at least four useful sharp locations and at most 13 ordinary control highlights. A useful location covers at least half the known changed pixels and reaches 0.30 intersection over union.

The 12 confirmation source photos are disjoint from the earlier boundary and point-setting groups. They have wider project exposure. Six contain patches derived from SDXL and six from Kandinsky. Each source contributes one sharp and one feathered insertion, plus six ordinary controls reused by the earlier production comparison. Match each of those 96 PNG views and its truth mask by digest before comparing results. The new hard-seam controls paste an unresized RGB patch from elsewhere in the same source and from the next genuine source photo into the same destination. They add 24 entirely conventional PNGs with exact changed-pixel masks. Their highlights show whether the observation also responds to ordinary compositing. [Photo terms and source separation](image-wavelet-region-transfer-2026-11-11.md) · [Frozen production-route observations](image-active-png-transfer-2026-11-11.md)

## Local results

| PNG views | Sharp boundary useful / highlighted | Current recompression useful / highlighted | Optional regional useful / highlighted |
| --- | ---: | ---: | ---: |
| Sharp generated inserts, 12 | 7 / 7 | 0 / 1 | 1 / 10 |
| Feathered generated inserts, 12 | 0 / 0 | 0 / 1 | 1 / 9 |
| Ordinary controls, 72 | — / 1 | — / 12 | — / 62 |
| Untouched originals, 12 of the 72 controls | — / 0 | — / 1 | — / 10 |
| Conventional same-image pastes, 12 | 8 / 8 | — | — |
| Conventional other-photo pastes, 12 | 9 / 9 | — | — |

The fixed check met both advancement criteria. Its one ordinary highlight was a local contrast adjustment. Both conventional paste groups produced useful locations, and the check missed every feathered edit. The result supports an inspection cue for abrupt rectangular seams in the tested PNGs. The source of a seam cannot be inferred from that cue.

The runner scored all 120 views locally without loading any model. Independent arithmetic checked all source-encoded views, truth masks, and region overlaps. It verified all 96 archived comparisons by exact PNG and truth-mask digests. The timing wrapper reported 11.34 seconds, then exited because the local sandbox rejected a host statistics query. A clean rerun reproduced the score journal and the audited aggregate byte for byte. The experiment makes no change to serving. [Reproduction runner](../backend/scripts/image_boundary_prominence_holdout.py)

From `backend`, invoke `python -m scripts.image_boundary_prominence_holdout --help` for the staged commands. The runner expects the prior qualified intake folder, the exact earlier points file, the saved active-PNG scoring folder and an existing empty output directory. The result retains aggregate measurements and whole-artifact hashes. Source images, individual proposed rectangles and fingerprints remain local.
