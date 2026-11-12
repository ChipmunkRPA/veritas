# Longer moving-repeat comparison on the native video clock

This study compares the current nine-second scope with an 18-second search on the same 640 × 360 H.264 and VP9 profile. The existing exact repeat and nine-second motion matches have priority. The longer comparison uses the same appearance and motion measurements on the first 450 verified native pictures at 25 fps. A timestamp gap anywhere in that scope leaves the opening result available. This is an inspection of repeated moving footage and carries zero AI-confidence weight. [Earlier 360p qualification](video-motion-sequence-360p-transfer-2026-11-12.md)

## Frozen comparison

Six previously unused Commons sources supply 18-second excerpts of animal play, rainwater, rail movement, grain grinding, sprinting and circuit training. Three edited camera controls join independent nine-second excerpts from six retained VCD2 camera files. These camera recordings appeared in earlier project tests. Exact bytes, source credits, extraction offsets and authored roles were fixed before scoring. The Commons files and all media stay local. [Publisher credits and case ledger](evaluation-panels/video-motion-late-transfer-2026-11-12.json) · [VCD2 data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

Each source has two late copies, one with occurrences on opposite sides of nine seconds and one with both occurrences later. Two other copies are entirely inside nine seconds, including one followed by an authored 8 ms clock step after frame 309. Ordinary controls include the unchanged source excerpt, a local cut to flipped footage, a static hold and a reverse. The matched conditions use one H.264 MP4 and one VP9 WebM delivery each. The initial direct MP4 encode lost its last frame in the presentation readback and stopped before scoring. Matroska encoding followed by an MP4 stream copy preserved all planned native timestamps.

A correct pair needs at least half overlap on each authored interval. The frozen gate requires eight correct late copies from the six published sources per codec, including at least three of six in each placement, and at least three of six late copies in the camera assemblies. It allows at most one added ordinary alert per codec and zero new held-picture alerts. Every useful existing result remains available. Ordinary loops and repeated edits can produce legitimate matches.

## Held results

H.264 MP4 and VP9 WebM produced the same counts across 72 deliveries per codec. The nine-second comparison below includes the existing exact-repeat check.

| Result per codec | Nine-second scope | Extended scope |
| --- | ---: | ---: |
| Published footage, copy straddles nine seconds | 0/6 | 5/6 |
| Published footage, both occurrences after nine seconds | 0/6 | 5/6 |
| Camera assemblies, copy straddles nine seconds | 0/3 | 2/3 |
| Camera assemblies, both occurrences after nine seconds | 0/3 | 3/3 |
| Copies within nine seconds, continuous clock | 9/9 | 9/9 |
| Copies within nine seconds, later clock gap | 9/9 | 9/9 |
| Ordinary controls with a repeat alert | 4/36 | 4/36 |

The extension localized 15/18 late copies per codec and added zero ordinary or held-picture alerts. All four existing repeat alerts came from the exact-repeat check on sprint footage, which contained recurring pictures. Exact-match precedence also prevented approximate reporting on the two sprint late copies. One cross-prefix camera copy fell outside the matching criteria. The independent replay checked all 144 delivery files, 64,800 native pictures and 12 unique source files. The slowest full native decode and description took 2.703 seconds on this Mac. The slowest motion comparison took 34 ms. [Results and replay receipt](evaluation-panels/video-motion-late-transfer-2026-11-12.json)

## Analysis and reports

The app runs the longer comparison when the existing exact and opening motion checks complete without a match, the original has 450 continuous verified 640 × 360 pictures at 25 fps in the tested H.264 or VP9 profile, and the worker has time left. Other cases retain the existing review scope. A qualifying repeat supplies the source times and pictures at both endpoints of each occurrence. The signed package can be checked again against the original video. Matches can result from ordinary editing. This panel measures copied motion within the checked span and leaves the overall AI-confidence calculation unchanged.

The local Full analysis flow verified the same two ranges in the API, JSON and standard PDF for both H.264 and VP9 of the held animal-play example. The ranges were 1.680–2.560 s and 11.520–12.400 s, with the four endpoint pictures at native frame indices 42, 64, 288 and 310. The plain browser view showed all four source pictures and sought to 11.520 s when the second range was selected. The local worker completed its 450-picture review on the unedited control and kept the opening 225-picture result for the control with a later clock gap.

## Reproduce

Supply the six Commons original files listed in the public ledger as `cat.webm`, `rain.webm`, `train.webm`, `grain.webm`, `sprint.webm` and `training.webm`. Use the retained v197 camera archive and a new local run directory. This comparison needs FFmpeg with H.264 and VP9 support and the project Python environment. No model weights are used by either repeat comparison.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_motion_late_transfer "$run" --publishers "$publishers" --archive "$archive"
backend/.venv/bin/python -B -m scripts.video_motion_late_transfer "$run"
backend/.venv/bin/python -B -m scripts.review_video_motion_late_transfer "$receipt" --run "$run"
python3 -B -S -m scripts.review_video_motion_late_transfer "$receipt"
```
