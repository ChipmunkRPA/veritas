# Full-expression frame bank completes on all pilot clips

The published CPU procedure completed all **16 fixed pilot clips**, producing
**640 planned condition frames and 16 exact fixed repeats**. Every generated
native composite changed pixels only within its intended mouth support. No
frame was skipped or replaced; all 320 pairs of source-time and earlier-driven
generated crops differed. This measures the generator's pixel response.

All workers exited zero with no resource guard violations or monitor sampling
errors. The largest sampled descendant RSS was **2.51 GiB** under the unchanged
6-GiB limit. The 16 stages took 37.61 minutes of summed worker-stage time,
139.20–145.04 seconds per clip. This was benchmark generation time; it does not
measure customer analysis latency. Samples do not guarantee the true memory peak.
See the [complete actual receipt](evaluation-panels/grid-portrait-frames-complete-v146.json)
and [frozen method](grid-portrait-frames-v146-method.md).

The actual journals reconcile 3,280 top-level generator forwards, 24,272 bounded
spatial convolutions and 155,472 functional tiles. The workers checked PNG pixel
round trips and native mask containment. The separate aggregation checked all
receipt, resource, root and operator journal commitments and counts without
re-decoding the images. The [initial two-clip receipt](evaluation-panels/grid-portrait-frames-initial-progress-v146.json)
remains the historical progress snapshot.

The reserve sources remain unscored. Achieved visible motion, phonetic alignment,
detector accuracy and comparison with the serving pipeline require their own
tests. No model or confidence changed in the app. Only aggregate outcomes and
whole-artifact commitments are public; media, per-input clocks, fingerprints,
face geometry and learned records remain private.
