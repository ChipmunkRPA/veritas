# Mouth-region holds in HD video

This study tests mouth-region picture holds on the first 120 native pictures of upright 1280 × 720 H.264 MP4 at exactly 30 fps with 8-bit 4:2:0 chroma. The comparison covers four-second and eight-second uploads. It measures local picture changes and carries zero weight in AI confidence. [Earlier short-video check](video-mouth-local-hold-2026-11-12.md)

## Frozen comparison

Eleven newly sourced Wikitongues recordings supplied distinct speakers. Two speakers supplied development cases. Seven other speakers supplied the held paired edits and controls. Two speakers whose published files used lower source cadences supplied ordinary controls only. Each file has an explicit commercial-use-compatible Creative Commons license. The case ledger records credits, file checksums, source rates, and the exact source excerpt offsets. The published files and derived media stay local. [Publisher collection](https://commons.wikimedia.org/wiki/Category:Videos_donated_by_Wikitongues) · [Per-file source and license ledger](evaluation-panels/video-mouth-hd-hold-2026-11-12.json)

The preparation finds the first qualifying offset from a fixed list. Each selected excerpt has one visible face in the relevant original picture samples and a 200 ms low-level interval bounded by higher audio levels. Eight of the nine held originals also have a qualifying 300 ms interval. These source-only checks do not use any mouth-hold score. Preparation converts frame rates to exactly 30 fps, resizes where needed, and independently checks the delivered picture clocks. Low source-cadence conversions provide additional controls for ordinary repeated pictures.

Each held edit group has a nine-picture hold from **1.167–1.467 s** in a four-second clip and a twelve-picture hold from **3.300–3.700 s** in an eight-second clip. Both use a single earlier picture to hold a fixed region with a five-pixel feather. The author checks actual changed source pixels on every edited picture. Controls include an unedited recording with its natural quiet interval, tandem H.264 compression, a whole-picture hold, a whole-picture cut, a local mouth blur, a held patch above the mouth, and a local mouth hold after the inspected prefix. The two low-cadence groups supply the first two ordinary controls.

The check applies the earlier native brightness, texture and neighborhood rules to three overlapping 60-picture windows, starting at frames 0, 30 and 60. It uses an 8–24 picture duration range and a five-picture flank at 30 fps. A useful result has at least 0.50 time intersection over union and at least 75% of its selected box inside the authored fixed core. The gate requires useful ranges on at least 75% of edited clips, useful late locations in at least four groups, at least 80% assessed coverage in each of the edited and ordinary cohorts, at most one ordinary alert, and zero alerts on cuts, whole-picture holds and edits after the prefix.

## Held results

| Held case | Outcome |
| --- | ---: |
| Early hold in the four-second upload | 6/7 localized |
| Later hold in the eight-second upload | 6/7 localized |
| Ordinary controls | 0/53 alerts |
| Deliveries with at least one assessed window | 67/67 |

Eleven of the 12 useful ranges match both authored endpoints. The remaining range overlaps 11/12 of its authored interval. One late case had usable geometry only in the opening window, before the edit. One early case had three assessed windows and no qualifying hold. The test flagged none of the nine genuine quiet intervals, nine tandem encodes or seven edits wholly after the four-second prefix. All 53 controls had at least one assessed window. Of the full 67-file panel, 52 cases assessed all three windows, seven assessed two and eight assessed one.

The largest recorded decode-plus-measurement time was 2.595 seconds under the local study load. Two groups were resumed serially on the Seagate archive after an OpenCV initialization error and temporary local disk exhaustion. The continuation used the same frozen method, source files, placements and gate. A separate readback verified 67 deliveries, 8,040 native pictures, 36 exact context crops and 113 measured transitions. [Held outcomes and native replay](evaluation-panels/video-mouth-hd-hold-2026-11-12.json)

In the app, the expanded check requires the verified 120-picture prefix. The browser and PDF list the windows with measurable mouth geometry. The signed JSON binds the selected range and its exact before, during and after frames to the native picture clock. Footage beyond that prefix falls outside this check. The result describes a stable region and local brightness change. Compression and ordinary editing can produce the same pattern.

In an isolated local Full analysis, the held eight-second example reproduced the exact **3.300–3.700 s** range with source frames **98, 105 and 111**. The browser navigated to the during frame, and the Standard PDF and JSON preserved all three source pictures and mouth crops. Signed replay verified the native inputs and pixel measurements against the original video. The genuine pause and the edit after the prefix both returned no qualifying hold. Three simulated paid analyses completed, with zero jobs queued before payment. [Local browser and export receipt](evaluation-panels/video-mouth-hd-hold-2026-11-12.json)

## Replay

Download the reviewed original files by following the ledger links and save each as the recorded source name followed by `.webm`. Supply that folder and an unused local run directory. FFmpeg, Python and the existing local face geometry asset are required for native replay. The public receipt also supports a standard-library arithmetic replay without media.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare development held report; do
  backend/.venv/bin/python -B -m scripts.video_mouth_hd_panel "$phase" "$run" "$sources" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_hd_panel "$run/receipt.json" --run "$run" --sources "$sources"
python3 -B -S -m scripts.review_video_mouth_hd_panel docs/evaluation-panels/video-mouth-hd-hold-2026-11-12.json
```
