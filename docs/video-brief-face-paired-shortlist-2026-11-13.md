# Scheduling both ends of a brief face change

The local app shows the six source frames compared at each end of a
bounded face change. The lists appear beside the timestamp range and
before, during, and after images in Media Evidence and in the Standard PDF.
They match the saved JSON at `/packets/{index}/face_return/edges/{entry|exit}/frame_indices`.
The comparison uses the first three and last three frames at each
boundary. Its result carries zero weight in the AI estimate.

Two H.264 encodes of a known edit passed the local report and signed
original-file replay. Both reported **2.067–2.267 s**, with an exclusive
end. The start comparison used frames **59–64**, the end comparison used
**65–70**, and the report used frames **61, 65 and 68** as the before,
during and after examples. An unchanged clip and an ordinary camera cut
added no brief face-change range. All eight runs with the option off
and on used simulated prepaid checkout.

## Detector experiment

The existing proposal ranks individual changes in nearby pixels. The
candidate gives first priority to pairs of shortlisted motion responses
five to seven pictures apart, ranked by the weaker response of each pair.
Both schedules keep the same 16-edge and 84-face-read budgets, existing
face measurements, response point and overlap rules. The candidate is
evaluated as an addition after the existing route supplies no short range.
The test verifies each decoded picture against its frozen clock and RGB
hash before measuring it. The two schedules share offline decoding.
[Qualified app inputs](video-brief-face-avc-clock-confirmation-2026-11-13.md)
· [Face assets and licenses](video-appearance-plan-v107-method.md)

| Outcome | Existing review | Existing review plus paired shortlist |
| --- | ---: | ---: |
| Useful matches to six-picture edits | 9/16 | 9/16 |
| Useful matches to eight-picture edits | 12/16 | 12/16 |
| Ordinary deliveries with any prior or new face-appearance range | 15/40 | 15/40 |

All 56 H.264 deliveries came from retained source groups. Each of 16
edited deliveries has two known intervals, including re-encodes of the
same intervals. The ordinary panel includes untouched clips, cuts within
one recording, cuts across recordings and genuine speaker changes.
A face-appearance range on these controls is a review observation.
The alternative changed the shortlist membership on four deliveries and
produced the same four exact new six-picture ranges as the existing
optional H.264 proposal. The tested fallback added zero useful ranges
and zero ordinary flags.

Every one of the 11 missed edits occurred in a delivery with an existing
short range, so the fallback routing rules excluded all 11. Four of
those edits are eight pictures long and are outside the five- to
seven-picture proposal duration. This reused panel measures the
limits of the fallback. It leaves open whether a second independent
range in an already flagged clip could recover the other edits. The
alternative shortlist remains outside the app.
[Replay receipt with native frame and result commitments](evaluation-panels/video-brief-face-paired-shortlist-2026-11-13.json)

## Replay

Recount the recorded decisions and check the native timestamps with the
Python standard library.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.video_brief_face_paired_shortlist review \
  docs/evaluation-panels/video-brief-face-paired-shortlist-2026-11-13.json
```

For a native rerun, restore both source banks and pinned face assets
listed in the [carrier study](video-brief-face-120-carrier-2026-11-13.md).
Use a new private path for each run.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_brief_face_paired_shortlist freeze "$run"
backend/.venv/bin/python -B -m scripts.video_brief_face_paired_shortlist run "$run"
backend/.venv/bin/python -B -S -m scripts.video_brief_face_paired_shortlist review "$run"
```
