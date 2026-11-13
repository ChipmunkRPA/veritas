# Matched speaker and layout cuts in genuine webcam footage

An ordinary video edit can replace a speaker, change the layout around a
speaker, or do both. This comparison measures which of those changes the
existing shot detector and off-face cut check register. Shot-boundary research
motivates treating same-layout cuts separately from broad picture changes.
[Shot-boundary research](https://arxiv.org/abs/2008.04838) ·
[Earlier off-face context](video-spatial-cut-context-transfer-2026-11-12.md)

## Source separation and controls

Eight genuine CREMA-D actors form four pairs. The freeze selected the first
eligible confirmation recordings in acquisition order, after excluding all
20 actor keys in the two previous CREMA-D scene-context banks. It verified
the cached publisher bytes and original decoded picture commitments before
using each source. These recordings had earlier project exposure. The actor
keys establish separation from the two named cut banks. The camera sources
and actor identifiers stay local. [Source preparation](video-common-source-preparation-v89.md) ·
[Publisher terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eebd53610a08b1dc/LICENSE.txt)

Each pair was tested in both directions. Each direction has four versions
with identical timing and delivery settings. The unchanged version has no
intervention. The speaker-only version changes the entire webcam rectangle
and keeps the surrounding layout. The scene-only version changes the layout
around the webcam and keeps every pixel within the speaker rectangle before
encoding. The fourth changes both. The two geometric layouts are deterministic
and use the same palette. The fixture checks changed and unchanged pixels on
each of the 3,840 rendered frames before encoding.

The output is silent 1280 × 720 HEVC `hvc1` at exact 30 fps, with 120 pictures.
It uses nearest existing pictures from the native camera clock and reflection
at the ends where needed to fill the fixed four-second delivery. The exact
edit intervals are **44–55**, **52–65**, **61–77** and **73–91**, each with an
exclusive end. No blended or generated speech pictures are used.

## Comparison

The existing whole-picture HSV shot detector runs on the complete native
delivery. Each boundary counts separately. The off-face comparison uses the
serving texture calculation and the fixed 0.30 point. It receives each
authored interval, or the same sham interval on the unchanged match. This
measures how the texture cue behaves when it has a range to inspect. A Full
analysis supplies its own face range before the context cue can run.

## Results

| Ordinary delivery | Files | Exact boundary pairs from the HSV shot check | Exact ranges from the off-face check |
| --- | ---: | ---: | ---: |
| Unchanged with a sham interval | 8 | 0 | 0 |
| Speaker rectangle changes | 8 | 0 | 6 |
| Surrounding layout changes | 8 | 0 | 8 |
| Speaker rectangle and surrounding layout change | 8 | 0 | 8 |

The off-face check obtained all three planned face observations in each
delivery. It reported no misplaced ranges. It found both speaker-only
directions in two source pairs and one direction in each of the other two.
One missed speaker-only view had no candidate. The other returned the exact
range with a response of 0.218803, below the fixed point. The whole-picture
check recorded no boundary in any delivery. All four authored cut durations
are short, and the layouts have a common palette.

The first scene-only example reported **44/30–55/30 s**. The 1/15360 source
clock bound its before, during and after pictures to indices **43, 49 and
55**, at exact ticks **22016, 25088 and 28160**. Independent readback
reopened the exact clocks for **3,840 delivered pictures** and re-decoded all
**96 representative pictures**. The scene-only case shows the off-face
cue can respond to a conventional layout cut even with identical speaker
pixels before compression. A customer result should treat that cue as
context for a possible edit.

## Full worker with unmarked uploads

The same 32 videos passed through the production Full face and tracking
pipeline in a private local database, with the cut-context option enabled.
No authored range entered the upload or the worker. Simulated test-mode
payment queued 32 analyses. The unpaid events queued zero. The local run
took 248.43 seconds. [Full worker decision ledger](evaluation-panels/video-scene-only-cut-full-2026-11-13.json)

| Ordinary delivery | Existing coarse face range | Exact cut context published |
| --- | ---: | ---: |
| Unchanged | 0/8 | 0/8 |
| Speaker rectangle changes | 8/8 | 6/8 |
| Surrounding layout changes | 0/8 | 0/8 |
| Speaker rectangle and surrounding layout change | 8/8 | 8/8 |

The eight layout-only videos reached neither the whole-picture shot check's
boundary rule nor the coarse face range needed to start the off-face check.
This accounts for the difference from the supplied-range comparison, which
located all eight. The worker checked the timestamps and exact original
pixels of all 42 published cut frames across 14 deliveries. The same evidence
appeared in the API and JSON export and carried zero overall-confidence weight.
The test used the existing opt-in path. A separate standalone scene timeline
would need to propose and verify cuts without a supplied face range on a new
held bank.

## Local replay

Use a fresh private directory and the retained source and actor-exclusion
manifests. The standard-library replay recomputes the decision totals and
checks the selected presentation-time commitments from the public ledger.
The source replay also reopens and verifies the actual pixels with FFmpeg.
[Decision ledger](evaluation-panels/video-scene-only-cut-transfer-2026-11-13.json)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare score; do
  backend/.venv/bin/python -B -m scripts.video_scene_only_cut_transfer "$phase" "$run" || exit 1
done
python3 -B -IS backend/scripts/review_video_scene_only_cut_transfer.py "$receipt" --assemble "$run"
OMP_NUM_THREADS=2 backend/.venv/bin/python -B -m scripts.video_scene_only_cut_full "$run" "$paid_run"
python3 -B -IS backend/scripts/review_video_scene_only_cut_transfer.py "$receipt" --full "$paid_run/receipt.json"
```
