# Scene continuity for short face-swap ranges

The earlier three-codec study found the same two false face-change ranges at genuine scene cuts. This experiment tests whether a stricter comparison of the background immediately before and after each boundary can suppress those kinds of false ranges while retaining real six-picture swaps. [Prior diagnostic](video-face-six-codec-transfer-2026-11-12.md) · [Frozen results and audit](evaluation-panels/video-face-scene-continuity-2026-11-12.json)

## Comparison

The two candidates review only a dense fallback range the existing check already selected. They use the two existing sampled boundary pairs and exclude tiles that intersect the observed face. The first candidate requires the lower of the two median tile correlations to reach **.83**. The second requires the lower of the two lower-quartile tile correlations to reach **.60**. These points are rounded midpoints of the previous study's observed cut and swap envelopes. The test keeps the rank point, first-pass ranges, eligibility and candidate ordering fixed. A suppression removes that one range and cannot select a replacement. The benchmark uses directly observed faces and the serving short-range calculation. Its coarse comparison bypasses full application tracking.

The roster was frozen before scoring. It has two development and six held recorded sources, all with the publisher's head-rotation tag. Documented original links and donor keys are disjoint between roles, among the eight sources, and from the 41 earlier short-localization groups. The private archive also contains these recordings for other project work, and missing portrait identity and pretraining links limit the separation claim. The source access and weight eligibility are inherited from the earlier qualified checks. [Source and rights scope](video-common-source-preparation-v89.md) · [Qualified model and delivery](video-face-six-picture-transfer-2026-11-12.md)

The experiment selects even decoded pictures on each source's measured 50 fps clock and scales them to 640 × 360. That preserves exact 40 ms source times without interpolating frames. All outputs are 120-frame, 25 fps H.264 with 4:2:0 chroma. The twelve held positives each insert six pictures from a publisher-provided face swap at **2.160–2.400 s** or **2.680–2.920 s**. This is temporal ground truth for the full inserted pictures. The experiment does not label which facial pixels were changed.

The held controls include twelve genuine cross-recording cuts, six genuine jumps within the same recording, and six each of unchanged, tandem-compressed, illumination and camera-punch-in deliveries. Pair selection compares coarse grayscale intensity in the outer part of three genuine source pictures. The closest held pair has a normalized distance of .023 and contributes four cuts matched by this measure across both directions and two durations. The remaining pairs have distances .286 and .399 and contribute eight cuts with distinct backgrounds. The coarse match measures picture similarity. The within-recording jump cuts supply controls from the same physical setting.

The frozen gate requires at least eight useful swap ranges across four held sources, zero loss of previously useful swap ranges, and removal of at least half of existing cut flags, with at least two removed cases across two cross-recording source pairs. A useful range overlaps at least half its temporal union with the authored interval.

## Held results

| Held outcome | Current | Median .83 | Lower quartile .60 |
| --- | ---: | ---: | ---: |
| Correct six-picture swap ranges | 10/12 | 10/12 | 10/12 |
| Correctly localized source groups | 5/6 | 5/6 | 5/6 |
| Flags on similar-background cross-recording cuts | 2/4 | 2/4 | 2/4 |
| Flags on distinct-background cross-recording cuts | 4/8 | 4/8 | 4/8 |
| Flags on same-recording jump cuts | 0/6 | 0/6 | 0/6 |
| Flags on other ordinary deliveries | 1/24 | 1/24 | 1/24 |
| New false ranges from the dense fallback | 0/42 | 0/42 | 0/42 |

Five useful ranges matched **2.160–2.400 s** exactly. Five selected **2.720–2.960 s**, 40 ms late at each endpoint. Both misses came from one source where the earlier short check found no eligible range candidate, leaving the dense fallback without a run on those views. All ten selected ranges cleared both stricter checks. The smaller edge correlation ranged from .993 to 1.000 for the median and .989 to 1.000 for the lower quartile.

The six false cut ranges came from the coarse research comparator across all three source pairs. Every one was **2.480–3.000 s**. The false flag on one stronger re-encode also came from the coarse comparator. The dense fallback selected zero false ranges in this panel, leaving either stricter check with zero removals and below the frozen benefit gate. The experiment left serving behavior unchanged.

An independent review reconstructed all 54 held MP4s byte for byte from 792 source pictures and verified 42 authored intervals, 6,480 encoded pictures, 40 selected boundary pictures, 30 face crops and all 162 decisions. The existing app short-range path separately reproduced **2.160–2.400 s** with native before, during and after anchors at **2.080**, **2.240** and **2.400 s**, and returned no range on its genuine control. Both app measurements ran in under 2.8 seconds. The held research run took 151 seconds.

## Existing shot check and remaining gap

The Full and Deep profiles already compare whole-picture HSV content and histogram changes on adjacent low-resolution frames and split face tracks at accepted shot boundaries. A retrospective readback of the unchanged shot stage found the exact entry on **8/8 distinct-background cuts** and **0/4 similar-background cuts**. It also marked the start of **2/6 illumination controls** and found no boundary on the **12 swaps**. The current half-second spacing rule keeps it from marking both ends of a 0.24- or 0.32-second cut. The earlier local application test found that tracked face review avoided all six false flags in a separate research comparison, so the present coarse research flags do not establish application false flags. [Earlier tracked application test](video-incumbent-cut-guard-transfer-2026-11-11.md)

A stronger coarse review should test both exact native boundaries for a brief scene departure and return. Useful evidence would show a broad change outside the face, stable pictures within the inserted segment, and continuity between the two flanks. That can distinguish a cut with a similar overall palette from a local face change. The next comparison should start with close-background cuts that actually produce a tracked coarse range, alongside genuine camera and lighting changes and known swaps.

## Reproduce

Use the retained native archive, including the pinned exposure ledgers and independent discovery audit. The final reviewer rebuilds held MP4s and reopens their native clocks and evidence.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare development held report; do
  backend/.venv/bin/python -B -m scripts.video_face_scene_continuity_transfer "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_scene_continuity "$run"
```

The retrospective shot-stage readback uses `app.detectors.video.shots.detect_shots(decode_stream(path, table), table)` on the same committed media and compares its boundary indices with each authored interval.
