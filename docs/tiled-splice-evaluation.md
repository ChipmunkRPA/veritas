# Small-edit tiling experiment

Completed locally on 5 November 2026. **Do not enable this prototype.** It did not
recover additional edits on fresh confirmation. [Aggregate measurements](tiled-splice-evaluation.json)
retain the failed approaches as well as source and decision commitments.

The experiment uses the existing pinned Apache-2.0 splice checkpoint, with no new
weights or training. Four overlapping tiles each cover 60% of the source width
and height. JPEG recompression differences are measured on the original full
image before cropping the feature planes, so tile boundaries do not introduce
a new JPEG compression grid. Eligible maps are projected back to native source
coordinates. Their internal six-percent border is suppressed; source borders
remain. Existing reliability exclusions and the two-region component cap apply.

The current selected map is retained whenever it displays a region. Only an
otherwise empty current result can receive the tile map. The prototype has no
weight in whole-file confidence. It is not registered in the customer service.

Two development hypotheses were separate:

- Global residual normalization added one false highlight and no edit overlap
  on 72 consumed original/edit pairs, in both native and JPEG conditions.
  It was rejected without consuming the new confirmation images.
- Tile-local residual normalization retained the same full-image recompression
  measurements but adjusted contrast within each tile. It recovered one extra
  native edit at one extra genuine-image highlight in development. JPEG gained
  no edit overlap. Gates were fixed using authentic development controls only,
  allowing one additional genuine-image review cue on that panel.

The second hypothesis advanced to 48 new source-image pairs: 16 each from Adobe
Firefly, SD2, and FLUX Fill, plus JPEG75 copies of both classes. Every earlier
TGIF source ID and the prior web examples were excluded before inference. The
nearest original-control DCT perceptual-hash distance to the prior 192 TGIF
originals was 16 of 63 bits; no exact byte overlap was found. This check cannot
prove independence from pretrained models or unknown near duplicates.

One published secondary prompt mask was missing. The fixed source was retained
and the missing mask recorded. Both original and edited images were required
and hashed; the primary edit mask remains their exact native RGB difference.
No case was replaced. Available author masks retained geometry/overlap checks.

| Held-out format | Current genuine highlights | Prototype genuine highlights | Current overlapping edits | Prototype overlapping edits | Mean edited IoU, both |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native PNG, 48 originals + 48 edits | 0 | 0 | 15 | 15 | 0.20384 |
| JPEG75, same 48 sources | 6 | 6 | 9 | 9 | 0.06082 |

All 192 source/format tile executions completed; each scheduled four tiles.
Both methods missed 33 native and 39 recompressed edits. The development gain
did not generalize. The original checks remain selected. These are localization
component outcomes, not headline AI decisions, confidence calibration, proof of
authorship, or an estimate of production-wide error rates. Formats share sources
and do not double the independent sample size.

Scripts are `tiled_splice_candidate.py`, `tiled_splice_localnorm_candidate.py`,
`score_tiled_splice.py`, and `review_tiled_splice.py` under `backend/scripts/`.
The source plan and raw maps remain in ignored local evaluation storage.
Development results alone cannot authorize adoption; the reviewer explicitly
requires a confirmation role for its local-adoption flag.
