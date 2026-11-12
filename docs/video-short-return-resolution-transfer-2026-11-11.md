# Short face-swap localization at 720p and 360p

The previously qualified short face-change check covered upright 1080p with an exact 25 or 30 Hz clock. This experiment tested the same range selection on native 720p25 recordings and paired 360p pictures. The 360p view uses area downsampling of each verified native RGB picture, so this arm measures a smaller picture grid without introducing a codec change. [Prior 1080p local serving verification](video-short-return-secondary-2026-11-11.md)

## Frozen comparison

The short check preserves the longer-window range and adds a distinct range only if the strongest same-shot candidate exceeds the existing 0.10 separation point, lasts under half a second, and overlaps the incumbent by less than 0.50 IoU. Localization requires at least 0.50 IoU against the labeled half-open edit. The longer-window comparator uses the existing triplet and window arithmetic with a research single-face stream. The deployed tracker can split tracks at scene cuts independently. [Deployed cut-guard verification](video-incumbent-cut-guard-transfer-2026-11-11.md)

Six archived RTFS groups use different known source links and donor keys from each other and from the preceding short-return and cut-guard panels. The freeze excludes confirmation-role and child-tagged clips. Each group supplies two swap families at two durations, an unchanged view, and matched light, blur and time-step edits at both durations. Three same-clock source pairs provide 12 genuine hard-cut controls. Each resolution sees the same source pictures and times. The nominal 0.375 s and 0.50 s labels round to 2.840–3.200 s and 2.760–3.280 s on the 25 Hz clock. The paired views share six sources. Source identities and media remain in the private local archive. [Native preparation](video-native-inserts-v95-method.md) · [Source and rights record](video-common-source-preparation-v89.md) · [Weight review](video-appearance-plan-v107-method.md)

Each view size must add at least six correctly localized shortest swaps across at least three source groups, assess at least 80% of swaps and controls separately, add at most two false ranges on the same-source controls, and add zero false cut ranges. The short check contributes review evidence and carries zero weight in the overall confidence score.

## Results

| Outcome | Native 720p, incumbent → combined | Derived 360p, incumbent → combined |
| --- | ---: | ---: |
| Short swaps correctly localized | 0/12 → 12/12 | 0/12 → 12/12 |
| Half-second swaps correctly localized | 12/12 → 12/12 | 12/12 → 12/12 |
| Same-source controls with false ranges | 0/42 → 0/42 | 3/42 → 3/42 |
| Ordinary source cuts with false ranges | 7/12 → 7/12 | 8/12 → 8/12 |
| Assessed by the new range check | 24/24 swaps, 54/54 controls | 24/24 swaps, 54/54 controls |

Both arms met the frozen gate with gains in all six groups and zero added false ranges. For the shortest edits, the native 720p incumbent produced no range on all 12. The derived 360p incumbent missed ten and placed two at the wrong time. The research incumbent's cut false alarms predated the secondary range. The CPU run verified 834 decoded source pictures in 135 seconds. The independent audit checked 312 native range pairs and reopened 72 before, during and after frame/crop anchors. Median overlap of the added correct ranges was 0.80 at each size. The largest timing error was one native frame, or 40 ms. The downsampled 360p result supports encoded replay, including compression and ordinary webcam controls. [Aggregate and independent source-crop audit](evaluation-panels/video-short-return-resolution-transfer-2026-11-11.json)

## Local application replay

The optional short-range review now also accepts upright native 720p at exact 25 Hz. The input still needs a 2–8 second native clock and enough single-face measurements. Enable the check with `VERITAS_VIDEO_FACE_REVIEW_MODE=appearance` and `VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW=true`. The 360p arm remains experimental.

Four 720p inputs ran through the local Full-profile checkout and worker with the face detection, tracking and appearance paths enabled. An isolated Stripe stub supplied test-mode responses. Checkout and an unpaid event created zero analyses. Each paid event queued one analysis and its repeated delivery added none. Every input ran with the short check off and on, and the fixture replay verified all 150 pictures and exact clock for each input.

On the selected positive fixture, the check flagged **3.000–3.360 s** for the known **2.920–3.280 s** edit, with 0.636 IoU. The other swap exercised duplicate suppression. The unchanged original and genuine hard cut had no face appearance range. The worker produced identical confidence decisions and incumbent ranges with the short check off and on.

The JSON export matched the evidence API. The three frame/crop pairs used the exact uploaded source pixels and appeared in the PDF. The signed ZIP passed offline replay against the upload. An archived 1080p version 12 package passed source replay with this verifier. Use this release's offline verifier for 720p packets.

## Reproduce

Use the retained archive and a fresh private output directory. The independent reviewer recomputes the decisions, checks the exact ranges and reopens the original files to verify the selected frame and face-crop pixels. The original freeze binds the detector source at `1cf4f637c2f9c823764d98802f6f9f1e4acacbbc`. Fresh runs bind their own source revision.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_short_return_resolution_transfer freeze "$run_dir" --archive "$archive"
backend/.venv/bin/python -B -m scripts.video_short_return_resolution_transfer run "$run_dir" --archive "$archive"
backend/.venv/bin/python -B -m scripts.video_short_return_resolution_transfer readout "$run_dir" --archive "$archive"
backend/.venv/bin/python -B -m scripts.review_video_short_return_resolution "$run_dir" --archive "$archive" --receipts "$native_receipts"
```
