# Local picture areas in matched H.264 clips

In a completed local video analysis, choose **Compare with an earlier video**.
For two upright 720 × 480 H.264 4:2:0 clips with 40–100 pictures, exactly
matching relative presentation timestamps, and a uniform rate of 23–25 fps,
the comparison can mark a region that differs for a short interval. It
shows the corresponding pictures and exact native crops from before,
during, and after. The PDF and JSON include the same source ranges. An
ordinary edit can produce this finding. It carries no AI confidence.

The previous comparison estimated a global brightness adjustment from
separate frame percentiles. A large local edit could shift those percentiles
enough to obscure the real area or fail the shared-content check. The new
method fits brightness and contrast using corresponding picture locations
and repeatedly removes the largest deviations from the fit. It keeps the
same interval selector, operating point, and 18% frame area cap. This is
an independent implementation of paired spatial measurement and does not
require a model. [Reference patch correspondence research](https://proceedings.mlr.press/v306/yoon26e.html)
· [Earlier extent trial](video-reference-extent-2026-11-13.md)

## Local comparison

Eight CREMA-D source groups were held apart from the previous extent
comparison. Their recordings had prior project exposure. The test created
new short local grade and pixel-copy operations with exact native truth.
It encoded matching 52-picture carriers at 24 fps. No clips were chosen
based on the new comparison. One replay rebuilt the panel from the eight
pinned public downloads with an empty cache. All 48 delivered pairs
reproduced byte-for-byte in the comparison ledger. [CREMA-D license](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

| Held result | Prior extent | Component refinement | Robust brightness fit | Both |
| --- | ---: | ---: | ---: | ---: |
| Large regions located | 3/8 | 8/8 | 8/8 | 8/8 |
| Small regions located | 8/8 | 7/8 | 8/8 | 8/8 |
| Local texture copies located | 8/8 | 8/8 | 8/8 | 8/8 |
| Delivery-only flags | 1/16 | 1/16 | 0/16 | 0/16 |
| Delivery-only pairs assessed | 16/16 | 16/16 | 15/16 | 15/16 |
| Wrong-source pairs admitted | 0/8 | 0/8 | 0/8 | 0/8 |

The robust fit alone retained all baseline locations and raised median
spatial overlap on the large edits from **0.081 to 0.889**. It declined
one of eight globally graded controls, so the drop in ordinary flags
comes with reduced coverage. It also localized 14 of 17 known redraws
on a previously consumed diagnostic, compared with 13 for the prior
extent. On the previously timing-qualified example it kept the exact
interval and covered 17,343 of 18,572 changed middle-frame pixels
(93.4%). The held run measures observable editing on a narrow carrier.
It supplies no new single-upload accuracy or lip-sync estimate.
[Pinned source roster](evaluation-panels/video-paired-local-area-sources-2026-11-13.json)
· [Held measurements](evaluation-panels/video-paired-local-area-transfer-2026-11-13.json)

## Local use and replay

Set `VERITAS_VIDEO_REFERENCE_COMPARISON_ENABLED=true` and
`VERITAS_VIDEO_REFERENCE_LOCAL_AREA=true` on a development worker.
The optional marked mouth-timing comparison uses its own existing method.
The full offline verifier reruns the comparison and checks the exported JSON
and all assets. The separate native interval verifier checks the six source
crops, timestamps, and source hashes without running the region selection.
An isolated Chrome checkout with simulated $9.99 payment tested a held pair.
The browser and PDF showed the same interval and native crop as the API.
The downloaded JSON passed both offline verifiers.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
roster=docs/evaluation-panels/video-paired-local-area-sources-2026-11-13.json
for phase in freeze acquire held; do
  backend/.venv/bin/python -B -m scripts.video_reference_robust_transfer "$phase" "$roster" "$run" || exit 1
done
python3 -B -IS backend/scripts/review_video_reference_robust_transfer.py "$run/held.json"
backend/.venv/bin/python -B -m scripts.compare_video_references "$earlier" "$analyzed" "$export" --verify
python3 -B -IS backend/scripts/verify_native_video_interval.py "$analyzed" "$export" --earlier "$earlier"
```
