# Lip timing from mouth-opening ranks within a clip

The audio-derived mouth-opening estimate learned from one set of
speakers transferred poorly to short native-camera clips. This study
asks how many failures are due to timing coverage, the scoring signal,
or the selection of an interval. It then tests comparing the order of
mouth-opening values within each clip. A rank comparison can use the
shape of a mouth-opening sequence when the predicted and observed
ranges differ in scale.
[Earlier dense test](video-lip-dense-posteriors-2026-11-15.md) ·
[Time-varying audiovisual alignment research](https://arxiv.org/abs/2203.16224)

## Comparison and diagnosis

Use the same 20 ms acoustic predictions, 40 ms measurement steps,
recorded picture timestamps, waveform edits and search for one local
change with matching timing before and after it. Compare rescaling the
predictions between their 10th and 90th percentiles with ranking both
the predicted and observed sequences within the clip. The rank rule
gives ties their average rank. Each method keeps the exact observed
picture indices, without estimating new frames. Each point comes
from the same four ordinary calibration speakers. The tighter rule
requires a score above their largest control score. A useful result has at least
0.40 temporal intersection over union and estimates the signed shift
within 80 ms. The study reuses the earlier pinned weights.
[Source clocks and weight terms](video-lip-dense-posteriors-2026-11-15.md)

For diagnosis, give each check the exact authored edit interval and
signed displacement. Keep the strongest common flank offset, then
record the first unsupported condition. This diagnostic can show that
an interval was unmeasurable or scored too low even if the search had
known exactly where to look. Its exact known interval is separate from
the detector's tolerant overlap metric.

In the first 48 native-camera shifts, the earlier dense check lacked
a comparable observation on one side or within the known interval in
one case. The requested offset was outside the grid in 17. The two
flanks failed to prefer the same offset in nine, and the within-edit
observations failed to favor the expected change in four. A constant
offset fit at least as well in two. The remaining 15 were below the
frozen point. Giving this dense check the edit clock did not produce a
passing result. For the sparse check, 33 of those 48 lacked enough
observations at the exact edit clock.

The separate wider-grid experiment followed that diagnosis. It expands
absolute offsets from ±240 ms to ±400 ms and preserves the limits on
the *difference* between offsets. It tests the original dense score and
the rank score separately. The wider grid requires common picture
support for every tested offset. On the first native panel it reduced
out-of-grid cases for the earlier dense score from 17 to nine and still
located none of the 48 shifts. On the main panel it reduced the rank
check's useful locations from 12 to eight. Both wider variants found zero
useful shifts on the separately held actors.

## Local fallback

The local review keeps the earlier sparse finding whenever it has a
supported interval with complete before, during and after evidence.
Otherwise, it can add one interval based on ranks. At the tighter
point, the actual order of those two checks yields the following
results.

| Evaluation group | Correct intervals, sparse → fallback | Ordinary controls flagged, sparse → fallback | Constant-delay controls flagged, sparse → fallback |
| --- | ---: | ---: | ---: |
| Main GRID PCM | 9 → 18 / 128 | 2 → 2 / 192 | 5 → 5 / 128 |
| Separate GRID PCM | 4 → 5 / 32 | 0 → 0 / 48 | 3 → 3 / 32 |
| Separate GRID AAC, 16 kHz at 48 kb/s | 3 → 4 / 32 | 0 → 0 / 48 | 1 → 1 / 32 |
| First native 29.97 fps actors | 2 → 5 / 48 | 0 → 5 / 72 | 1 → 2 / 48 |
| Remaining native 29.97 fps actors | 2 → 4 / 42 | 0 → 0 / 63 | 0 → 1 / 42 |

The separate PCM and AAC deliveries share recordings. Both recovered
the same additional edit. The remaining native panel used the other
21 qualified actors from the source audit. It contains 147 total
deliveries, including two staged shifts per actor. The points stayed
fixed. The native results still vary between actor groups, and both
native and wide-offset variants stay in evaluation. The local option
uses the existing short single-shot scope and continuous 16 kHz mono
PCM or AAC soundtracks. The comparison describes a timing change and
carries zero weight in the AI assessment.
[CREMA-D source terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/master/README.md#license) ·
[Measurement receipt](evaluation-panels/video-lip-aperture-scale-2026-11-15.json)

## Local verification

Use `VERITAS_VIDEO_LIP_TIMING_RANKED=true` with the existing local Full
lip-timing setting and the same installed assets. The added check
reuses the same phone-decoder forward pass. The report follows the
normal timing evidence path and binds the waveform and native frames
to their source ticks.

The held AAC example was inconclusive under the sparse check and found
audio at **1.160–1.880 s** and corresponding pictures at
**0.920–1.640 s** with the additional check. The reported change was
approximately 280 ms later than the surrounding timing. The staged
change was 200 ms. Playback, the before, during and after mouth crops,
the API and JSON, and the PDF agree on the measured range. The signed
package verifier replayed the timing from committed measurements and
verified the pictures and waveform against the supplied video. Four
simulated USD 9.99 checkouts queued zero unpaid jobs. The ordinary
recording returned no finding, and a silent source was outside the
review. [Local verification receipt](evaluation-panels/video-lip-aperture-app-2026-11-15.json)

## Replay

The standard-library replay reconstructs both points and the GRID
results from the existing source measurements. The native section can
also be replayed after regenerating the two actor receipts. The public
package contains the aggregate and file digests. Source pictures, clips
and detailed actor measurements remain local.

```sh
export PYTHONPATH=backend
module=scripts.video_lip_aperture_scale_transfer
receipt=docs/evaluation-panels/video-lip-aperture-scale-2026-11-15.json
python3 -B -S -m "$module" review "$receipt"
python3 -B -S -m "$module" review "$receipt" \
  --audit "$native_audit" --previous "$first_actors" --remaining "$held_actors"
```

To produce the native receipts, use the [earlier native capture command](video-lip-dense-posteriors-2026-11-15.md#replay).
Use `--remaining-actors` for the second capture. To run the local app
fixture with the separate AAC carrier panel and the prior assets:

```sh
backend/.venv/bin/python -B -m scripts.video_lip_timing_app "$run/app" "$lombard_archive" \
  docs/evaluation-panels/video-lip-viseme-carrier-2026-11-15.json.gz \
  "$checkpoint" "$landmarker" "$runtime" --ranked --standard-checks
```
