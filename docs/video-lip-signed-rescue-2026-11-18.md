# A temporary change between speech and mouth timing

The test measures sound that moves earlier or later for a short part of one
continuous speaking shot. It compares an outside lag and a possible changed
lag using exactly the same measurable frames at both offsets. The outside
lag must give a stronger comparison on each flank. The changed lag must give
a stronger comparison inside. The weakest of those three differences sets
the response. The check maps the proposed interval onto the submitted audio
clock using the inside lag. The lag difference gives a direction and size.

The two representations and landmark assets come from the existing local
project. The earlier whole-shot timing comparison used different fitted
features and found no useful signed location on 20 held edits. Frame-level
research also measures sound and lip timing in a shared representation.
[Prior temporary-lag study](video-lip-offset-path-2026-11-14.md) ·
[Existing locally trained assets](video-mouth-and-sound-2026-11-17.md) ·
[Research on temporally aligning speech and lips](https://arxiv.org/abs/1808.06250) ·
[Frame-level research](https://arxiv.org/abs/2610.30872)

## Frozen comparison

The verified Lombard Grid archives supply 18 distinct recordings from 18
speakers. Six speakers set the operating point, and 12 different speakers
supply the held comparisons. The source selection excludes the recorded
utterances in earlier video source rosters and the speakers used to fit or
select the two-stream representations. Some speakers have other earlier
project uses. The study uses the plain recordings and authored local sound
edits. It covers one visible speaker in a controlled capture setup.
[Publisher archive and CC BY 4.0 terms](https://zenodo.org/records/3736465)

Each held source supplies six MP4s. Keep the original speech, change its
level for a short interval, move the entire soundtrack earlier or later, or
apply those same signed movements within the selected interval only. The
local edits use 640 or 880 ms and move audio by 200 or 280 ms. Their two
edges blend over 8 ms. Before encoding, samples outside the edited interval
remain byte-identical. Every file preserves its native picture stream and
presentation times. The input is one continuous 48 kHz mono AAC track,
reopened for inference from the same MP4 as the picture stream. The two
hypotheses use the re-decoded 16 kHz samples and unchanged native pictures.

The picture timestamps have zero origin and one exact cadence per source,
near 24 fps. An intake guard initially expected integer frame rates. The
measurement used the existing application clock bounds after the exact
rational rates were confirmed. The correction retained the ordered source
roster, all treatment labels, the decision rules and the comparison score.
The final calibration replay then set the point before the held run. An
independent picture read checks the stored frame hashes for each proposed
range. The AAC readback must align with the authored samples to within one
16 kHz analysis sample.

The baseline is the current local timing, articulation, near-silence and
learned-agreement evidence path. When it already has a range, it keeps
priority over the candidate. A correct location covers at least 40% of the
temporal union of the reported and labeled audio ranges. A useful signed
location also has the correct direction and is within 80 ms of the authored
movement. Calibration sets a point from the strongest measured ordinary
control plus a fixed allowance. The decision requires four new correct
signed locations across three speakers, with explicit limits on additional
ordinary, whole-track and misplaced ranges. [Frozen plan](evaluation-panels/video-lip-signed-rescue-plan-2026-11-18.json.gz)

## Held results

| Method | Assessed files | Matched locations on 24 local edits | Correct signed locations | Alerts on 24 original or gain controls | Alerts on 24 whole-track offsets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current local timing | 50/72 | 11 | 2 | 1 | 5 |
| Signed comparison | 72/72 | 3 | 3 | 2 | 1 |
| Current timing with signed fallback | 72/72 | 12 | 3 | 3 | 5 |

The fallback added one correctly located and signed range. It estimated
sound 200 ms earlier in 0.715–1.379 s, inside the authored 0.675–1.555 s
edit. The two additional ordinary alerts come from one speaker. All other
timing locations stayed clear of the misplacement definition. The fallback
fell short of four useful additions across three speakers and remains an
evaluation candidate.

Before applying the point, the strongest comparison overlapped 19 of the
24 local edits. Eleven also had the correct signed shift. Eight of those
eleven scores fell below the frozen point of 0.4639570. That diagnostic
identifies overlap between genuine and edited response strengths as a
remaining limitation on these clips. The held point is kept as measured.

All 96 media deliveries completed, including 24 calibration controls.
Readback verified 6,208 native picture timestamps and 273 proposed source
frames, and decoded AAC aligned at zero samples in every delivery. A second
calibration run reproduced all 24 measurements and the operating point.
[Calibration measurements](evaluation-panels/video-lip-signed-rescue-calibration-2026-11-18.json.gz) ·
[Held measurements and exact anchors](evaluation-panels/video-lip-signed-rescue-held-2026-11-18.json.gz)

## Replay

The numeric reader uses standard-library Python. The native replay uses the
official frontal video, audio, metadata and alignment archives. Use the
publisher paths and member identities in the frozen plan and the pinned
landmark, speech and local sound and mouth assets from the earlier study.
The local model and original media stay on the replay machine.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
panels=docs/evaluation-panels
plan="$panels/video-lip-signed-rescue-plan-2026-11-18.json.gz"
cal="$panels/video-lip-signed-rescue-calibration-2026-11-18.json.gz"
held="$panels/video-lip-signed-rescue-held-2026-11-18.json.gz"
python3 -B -S -m scripts.review_video_lip_signed_rescue "$plan" "$cal" "$held"

runner=backend/.venv/bin/python
"$runner" -B -m scripts.video_lip_signed_rescue stage \
  "$plan" "$sources" "$models" "$timing_assets" "$work/calibration" --phase calibration
"$runner" -B -m scripts.video_lip_signed_rescue stage \
  "$plan" "$sources" "$models" "$timing_assets" "$work/held" --phase held \
  --calibration "$work/calibration/receipt.json.gz"
```
