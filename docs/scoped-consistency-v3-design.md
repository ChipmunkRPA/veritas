# Scoped consistency for separately calibrated evidence

This research alternative applies to images, video, audio and passages. It
does not change the serving application. Correct arithmetic is not evidence
of better detection: real source-disjoint comparisons are still required.

Every planned check retains its own raw response, independently fitted local
calibration, semantic target, exact source location and availability status.
Four connected-source roles separate specialist calibration, combination
fitting, combination calibration and confirmation. The support floor remains
25 source groups per class. Customer model identities remain private.

The previous combination keeps the largest positive calibrated logit in each
dependency family. This prevents a genuine area elsewhere from cancelling a
suspicious area. But it loses the difference between two views agreeing about
one area and one positive view contradicted by another view of that same area.

The new alternative supplies two fixed features per dependency family:

1. The strongest positive indication in any planned source unit.
2. The strongest positive minimum among distinct views sharing the **exact
   target and exact source location**, when at least two such views exist.

For example, same-unit logits (+4,+4) give features (4,4); (+4,-4) give (4,0).
Positive checks at different timestamps or passages cannot create same-unit
agreement. A negative different area cannot erase the strongest indication.
Duplicate observations of one specialist cannot manufacture consistency.
With one measured view, consistency is zero; with a missing planned response,
the entire required combination is unavailable rather than filled with zero.

These are inputs to a separately fitted monotone combination and another
calibration. They are not independent votes, multiplied probabilities, proof
of authorship, or validated customer accuracy. Identical model views may still
share biases. Same-location consistency can fail when both views are wrong;
real false-alarm and missed-edit comparisons must decide whether it helps.

Coordinates remain modality-specific: original stored pixels, measured video
PTS/frame ranges, original audio samples/sample rate, and offsets into exact
extracted text. Region presence does not imply a calibrated pixel heatmap.
Missing face detection or background/lip checks cannot become an original
file verdict. The original v2 module and all running frozen studies remain
byte-unchanged; version3 artifacts cannot be used interchangeably with them.

Sixteen authored tests exercise these properties across all four media. They
are software checks, not fresh-media performance results. The first real
model-free comparison will reuse consumed writing-v5 scalar responses and
report the original reference, shared-family maximum, and scoped alternative
separately. Audio-v15 and actual PNG/JPEG-v11 measurements remain frozen
under their original methods. A later scoped comparison must be explicit,
preserve their original results, and not relabel consumed sources as fresh.
