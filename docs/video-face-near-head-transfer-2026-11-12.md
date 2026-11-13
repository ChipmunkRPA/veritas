# Near-head context on video cutaways and face swaps

A previous test found that a band below the face often fell outside the image. It supplied no useful extra context on held speaker cutaways. This experiment brings the comparison closer to the head and measures it on different source groups. Both experiments carry zero AI-confidence weight. [Earlier subject-context comparison](video-face-subject-context-2026-11-12.md)

## Frozen source-timed transfer

The genuine controls use ten archived CREMA-D actors, paired into five distinct groups in archive order. The exact originals and native decoded RGB were independently qualified before this study. Each pair supplies a fixed-layout different-person cutaway, a tandem re-encode of that cutaway, and four matched sham or processing controls. Those controls preserve the same actor, with either no intervention, a local jump to an earlier moment, a digital punch-in, or a local brightness increase. The ordinary cuts switch the whole webcam tile. All files use a fixed textured surround. The 120 delivered H.264 pictures are at exactly 30 fps. The bounded rendering recipe indexes native pictures around the clip midpoint and reflects at the ends using existing pictures. The resulting controls are engineered layouts. [CREMA-D publisher terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Prior native source audit](native-mouth-clips-v120-complete.md)

The manipulated set uses eight already qualified RTFS groups with two exact swaps per file, each in a first encode and a tandem-compressed delivery. Host URLs, source groups and known donor keys are separate from all seven RTFS groups that informed the earlier context experiment. Both sets have prior project exposure. The dataset publisher declares CC BY-SA 4.0 and supplies incomplete donor attribution. Public records contain neither source media nor per-actor information. [Exact transfer inputs](video-face-two-returns-hd-2026-11-12.md) · [Source permission review](video-common-source-preparation-v89.md)

Starting from six native flank pictures, the check fits a median face rectangle from at least four compatible detections. It measures a central face ellipse and four nearby ring sectors. The outside area is substantially closer to the face than the earlier lower band. It needs two sufficiently visible sectors and an observable central transition at both boundaries. Each positive sector needs at least eight levels of excess luminance change and at least 55% of the central change at the same edge. The method subtracts ordinary adjacent-picture change estimated before and after each edge.

The trial supplies the exact authored boundaries, including matched sham times. It tests the cue under known timing. It does not estimate candidate timing or customer report coverage. The fixed gate requires four of five genuine cutaway groups to show broad context, at least 75% assessment separately in each of the genuine-cutaway, swap and ordinary-processing banks, at most two broad annotations on known swaps, and at most one on the ordinary processing controls.

## Results

| Delivery | Intervals | Spatial measurement available | Central transition at both ends | Broad change at both ends |
| --- | ---: | ---: | ---: | ---: |
| Genuine cutaway | 5 | 5 | 5 | 5 |
| Tandem genuine cutaway | 5 | 5 | 5 | 5 |
| Publisher swaps | 16 | 16 | 16 | 0 |
| Tandem publisher swaps | 16 | 16 | 15 | 0 |
| Unchanged layout | 5 | 5 | 0 | 0 |
| Same-person jump | 5 | 5 | 4 | 0 |
| Same-person punch-in | 5 | 5 | 5 | 5 |
| Brightness step | 5 | 5 | 5 | 1 |

The narrower region supplied coverage on all 62 source-timed intervals. It distinguished the five genuine cutaway groups from all 32 tested swap intervals in this assay, including tandem copies. It also annotated every digital punch-in and one brightness step. That is six broad annotations on 20 other ordinary controls, above the prespecified maximum of one. The candidate stays experimental. A useful follow-up would measure whether simple image registration or brightness normalization can account for the outer change before attributing it to an ordinary cut. Any candidate would need both new source groups and a test starting from the app's own selected range. [Aggregate and gate](evaluation-panels/video-face-near-head-transfer-2026-11-12.json)

The first scoring attempt stopped on a NumPy scalar during JSON encoding, after recording one unchanged sham. The second freeze changed only scalar conversion. The source roster, placement, formula and gate were identical. All thirty rebuilt control deliveries matched the first preparation's encoded file and native RGB digests. The completed transfer verified 1,426 source frame reads.

## Local replay

Use the private completed subject-context run, the archived CREMA-D originals and the exact 720p two-return study inputs. Use a new private working directory. The tool binds the two source studies, exact decoded picture digests, source clocks, native detector assets and frozen parameters.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze prepare score report; do
  backend/.venv/bin/python -B -m scripts.video_face_near_head_transfer "$phase" "$subject_context" "$run" || exit 1
done
```
