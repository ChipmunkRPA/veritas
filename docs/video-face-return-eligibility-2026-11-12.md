# Recovery checks for brief native face changes

The current tracked face review correctly localized 28 of 30 held swaps. Two changes to the search recovered zero additional ranges. Four ordinary recompressions drew existing coarse face alerts. None of the ten whole-picture cuts drew a range. Both new search rules stay experimental. [Frozen outcomes and independent review](evaluation-panels/video-face-return-eligibility-2026-11-12.json)

The current short-range check scans 8 Hz first and may scan every second native picture in a supported 25 fps delivery. Two coverage gaps follow from its selection order. The denser pass starts only after the first pass finds a guarded candidate and produces no distinct range. The candidate with the strongest response is also chosen before checking its duration and overlap with the coarse result. This experiment tests those two behaviors separately. [Current short-range app and evidence contract](video-face-six-duration-transfer-2026-11-12.md)

## Fixed comparison

- **Dense pass after unavailable context** runs the existing dense measurement when all first-pass native measurements completed but no window met its face and scene guards. Missing timestamps, multiple faces, incomplete decoding and a selected first-pass range retain their current handling. The two passes share the existing 12-second time budget.
- **Eligibility before ranking** considers only candidates within the existing duration, response and coarse overlap limits. It considers the measured sparse pass first and uses the measured dense pass if needed. A candidate can add a short range only if production selected none. All existing displayed ranges are preserved.

Neither candidate adjusts the internal response point or the face or scene checks. A useful location needs at least half temporal IoU with the known edit. The frozen criterion requires at least three new useful locations across two sources, at most one additional ordinary flag and no new cut flags. The experiments measure face appearance for inspection and add no confidence weight.

The source roster uses all ten remaining complete native groups outside the 77 documented source and donor groups in the preceding short-range and scene studies. Previous project or model-training exposure may still exist. Every output is 640 × 360 H.264 with 4:2:0 chroma and exactly 120 pictures at 25 fps. The held authoring recipes include two six-picture placements, a ten-picture placement, a medium swap, and ordinary recompression, illumination, same-source jumps, cross-recording cuts and unchanged originals. Every output tick takes the nearest authenticated native source picture. The original and publisher swap clocks were checked before placing edits. [Source and rights](video-common-source-preparation-v89.md) · [Previous source exclusions](video-native-background-challenge-2026-11-12.md)

A source-only preflight found an interval in which a publisher swap left at least one picture identical to the original. No detector had scored any new delivery. The new freeze checks both publisher variants at the named interval, then tries a two-picture shift in either direction. It keeps the same duration and sampling phase and accepts a span only if every source picture differs. Cases without exact source support are recorded as unavailable.

In a separate diagnostic on previously exposed clips, both sample grids had no guarded candidate on two short misses. Each edge had seven textured background tiles outside the face, one below the required eight. Before applying that scene requirement, the denser grid found one exact interval and one with five-sevenths temporal overlap. A longer missed swap returned negative separation on both grids. The held run uses different source groups and the real Full face and tracking stages.

## Held outcomes

| Delivery | Current useful locations or ordinary alerts | Added by denser retry | Added by eligibility-first ranking |
| --- | ---: | ---: | ---: |
| Six-picture swaps | 18/20 | 0 | 0 |
| Ten-picture swaps | 10/10 | 0 | 0 |
| Ordinary recompressions | 4/10 | 0 | 0 |
| Original, illumination and same-source jumps | 0/30 | 0 | 0 |
| Whole-picture cuts | 0/10 | 0 | 0 |

Each publisher variant supplies 15 of the 30 measured swaps. All ten planned medium swaps lacked exact changed pictures throughout every admissible interval, so this trial has no medium-swap result. Both six-picture misses came from one source. The denser grid was already measured for both. Neither grid contained an eligible range. The denser retry ran on the one completed sparse case without sufficient same-shot context and found the same coverage gap. The 79 other sparse passes already reached a localization decision. There were no recorded time-budget or incomplete measurement cases. The retry uses a 12-second cap from the start of the sparse pass and still observes the outer stage deadline. Production can reserve part of that outer deadline for the coarse result.

The four recompression alerts selected **2.080–2.880 s** twice, **1.320–1.840 s**, and **3.160–3.680 s**. Each corresponding original had zero face alerts. These are real coarse-stage false alerts on known ordinary deliveries. They motivate a separate compression test that measures preservation of the true swaps.

The Full-stage comparison completed 80 runs in 376 seconds. The independent review rebuilt all 80 MP4s byte for byte, checked 1,200 native original and 220 publisher swap pictures, and verified 363 coarse-stage input picture hashes and native timestamps. It rechecked all 80 outcome decisions from the saved ranges. Face and scene measurements came from the serving detectors.

## Replay

Use the retained native archive and the previous local challenge freeze. The output directory must be new. The direct screen uses the existing qualified CPU weights and the same Full face, tracking and decode stages as the preceding cut challenge.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare score report; do
  backend/.venv/bin/python -B -m scripts.video_face_return_eligibility "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_return_eligibility authoring "$run"
backend/.venv/bin/python -B -m scripts.review_video_face_return_eligibility outcomes "$run" --publication "$receipt"
```
