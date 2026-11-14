# Competing offsets in single-track video timing

New video reports show the strongest alternate timing offset at least
120 ms from the selected peak. The browser, PDF and JSON show its signed
time, correlation and the difference from the selected correlation. The
complete recorded search still has 31 offsets. The public verifier recomputes
the alternate offset from that profile. The correlation difference describes
how much separation the measurement provides and carries no additional
AI-confidence weight.

## Local comparison

The experimental rule used a 480, 640 or 880 ms candidate with two 480 ms
flanks. Both flanks had to favor the same reference offset, while the
middle had to favor a different offset. The first arm used a fixed
bracket score of 0.25. The second kept it and required each selected
correlation to beat every offset at least 120 ms away by 0.10 or more.
These are experimental correlation criteria in an uncalibrated comparison.
Timing and visible articulation are separate signals. This comparison
tests timing using the existing visual aperture proxy and speech RMS.
[Timing and articulation study](https://arxiv.org/abs/2610.09223)

The replay used 11 held acted recordings, 11 previously evaluated
conversation excerpts, and 10 new excerpts from five other Wikitongues
recordings at 68 and 80 seconds. Both source sets were previously acquired.
Five acted recordings were too short for the comparison and remain in the
planned denominator. Each input had two staged local soundtrack shifts
and eight controls including uniform delay, gain, filtering, noise, and
matched motion and audio. The matched controls reassign measured pictures
and audio to corresponding ticks. Useful means at least 0.40 temporal IoU
and signed lag within 80 ms of the planned shift.
[Earlier source intake and face-asset terms](video-lipsync-window-schedule-2026-11-14.md#frozen-comparison)
· [Transfer recordings and terms](video-short-soundtrack-2026-11-14.md#held-comparison)

| Cohort | Current useful / controls flagged | Fixed bracket | With distinct peaks |
| --- | --- | --- | --- |
| Acted | 0/22 · 0/88 | 1/22 · 10/88 | 0/22 · 6/88 |
| Earlier conversations | 0/22 · 0/88 | 5/22 · 70/88 | 1/22 · 30/88 |
| New excerpts | 0/20 · 0/80 | 2/20 · 21/80 | 1/20 · 8/80 |
| Total | 0/64 · 0/256 | 8/64 · 101/256 | 2/64 · 44/256 |

Peak separation reduced flags while removing six useful locations. The
guarded arm also flagged eight edited clips at the wrong time or lag
and retained errors on whole-track delays and synchronized edits. Both
proposals remain offline. These tests do not cover synchronized neural redraws
or estimate the false-alarm rate on natural video.
[Scalar source and decision ledger](evaluation-panels/video-lip-lag-prominence-2026-11-14.json)

For three of the five retained Wikitongues publisher videos, the original
Vorbis export decoded into five clock spans at the start. The last span
began at 0.2675 seconds and contained the rest of the checked PCM. The
largest decoded overlap was 2.5 ms. The timing detector still withholds
single-clock correlation on those original deliveries. This comparison
uses separately rendered excerpts, so it does not measure recovery from
timestamp discontinuities. [Earlier gap experiment](video-audio-clock-gap-transfer-2026-11-11.md)
· [Matroska timestamp units](https://www.rfc-editor.org/rfc/rfc9559.html#name-timestamp-ticks)

## Recompute the results and report

Use the pinned actor and conversation caches from the earlier study and
the five source files linked for the transfer. Use the same local FFmpeg
and OpenCV versions for byte-for-byte receipt comparison. The public
ledger contains positions, measurements and hashes. It contains no
audio or images.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipt=docs/evaluation-panels/video-lip-lag-prominence-2026-11-14.json
python3 -B -S -m scripts.video_lip_lag_prominence review "$receipt"
backend/.venv/bin/python -B -m scripts.video_lip_lag_prominence run "$replay" \
  --actors "$actor_cache" --conversations "$conversation_cache" --fresh "$public_cache"
cmp "$receipt" "$replay"
cd backend
.venv/bin/pytest -q tests/test_video_nested_evidence.py::test_competing_lag_reaches_pdf_json_and_source_replay
```
