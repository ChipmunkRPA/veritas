# Signed lip timing with within-shot normalization

This study tests ways to separate a brief speech delay or advance from
ordinary speech in one visible speaking shot. All four variants use the
same existing mouth and speech weights, and compare each pair of offsets
on the same observed frames. The held test applies the single rule selected
on development.

- **Earlier contrast** asks whether one offset explains the flanks and a
  different offset explains the middle.
- **Near-zero outside lag** requires the outside alignment to fall within
  60 ms of the native synchronization.
- **Normalized contrast** divides each frame's lag responses by their
  interquartile range, with a fixed floor, before comparing them. It limits
  the effect of frames where the representations give unusually large or
  small differences across candidate offsets.
- **Guarded normalization** combines the normalization and near-zero
  outside requirement. The same large displacement supported separately by
  both halves suppresses a temporary proposal.

The settings in the earlier signed study reached eleven correct signed
locations before its point, and three above the point. The strongest
ordinary input set the operating point. This follow-up uses those recordings
as development, including both earlier partitions.
[Earlier signed study](video-lip-signed-rescue-2026-11-18.md) ·
[Existing model and permission review](video-mouth-and-sound-2026-11-17.md) ·
[Time alignment research](https://arxiv.org/abs/1808.06250)

## Development and untouched holdout

The development run verifies all 96 earlier upload hashes and reopens
those clips. It adds two smooth, whole-track timing drifts for each of the
18 development recordings. The drifts change continuously between 18 ms
and 160 ms, one earlier and one later. Frozen selection maximizes correctly
signed additions across the development clips, followed by the contributing
speaker count and the fewest additional alerts. It admits up to two extra
ordinary flags, zero new offsets on whole-track or drift controls and up to
two wrong locations. Existing findings retain priority when the candidate
is added.

At their retained development points, the earlier contrast, near-zero
outside lag and guarded normalization each added one signed location with
zero new ordinary, uniform or drift alerts. Guarded normalization added one
further matched range without a correct signed estimate. Normalization alone
added zero signed locations. The fixed tie-break kept the earlier contrast
and set the held point to 0.5233954.

The held panel uses ten different recordings from ten different speakers.
The final roster excludes both earlier partitions and the speakers used to
fit or select the existing shared representations. It contains both plain
and Lombard speech. Each recording has an unchanged copy, a gain control,
two uniform offsets, two smooth drifts and two temporary direction edits.
The temporary edits last 600 or 840 ms and move sound by 200 or 280 ms. This
version uses the new authored boundaries only for labeling. The individual
checks receive one MP4, its picture timestamps and the decoded soundtrack.
[Publisher recordings and CC BY 4.0 terms](https://zenodo.org/records/3736465) ·
[Frozen source and method plan](evaluation-panels/video-lip-signed-contrast-plan-2026-11-18.json.gz)

All clips use their exact native H.264 pictures near 24 fps and mono
48 kHz AAC sound. The runner verifies the picture and sample clocks of
each input, then independently reopens the proposed source frames. A
location must overlap at least 40% of the temporal union with the known
audio edit. A signed location also needs the correct direction within
80 ms. The held adoption rule requires four new signed locations across
at least three speakers and applies the same limits on added alerts.

One planned local advance would need 2,080 audio samples past the end of
its publisher recording. It remains in the planned denominator as
unassessed. The correction retained the roster, planned positions, score
and decision rules. The rerun reproduced the 132 earlier development rows
and the 60 held rows completed before the intake stop exactly.

## Held results

| Method | Matched ranges on 20 planned edits | Correct signed estimates | Alerts on 20 original or gain controls | Alerts on 20 whole-track offsets | Alerts on 20 smooth drifts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing timing | 4 | 0 | 5 | 3 | 4 |
| Earlier contrast, prior point | 3 | 2 | 0 | 0 | 0 |
| Earlier contrast, selected point | 2 | 1 | 0 | 0 | 0 |
| Existing timing with selected fallback | 5 | 0 | 5 | 3 | 4 |

The selected point added one range with 0.72 temporal overlap and the
opposite timing direction. Its correct signed result occurred on an edit
with an existing unsigned range, which kept priority. The two misplaced
existing ranges persisted and the fallback added zero misplaced ranges.
The result fell short of the required signed additions and remains an
evaluation candidate.

Before applying the selected point, the strongest range overlapped 12
of the 20 planned edits and gave seven correct signs. It also produced a
proposal on every one of the 60 control files. This panel confirms that
the measured response strength still overlaps too much between ordinary
and edited inputs.

The 132 development and 79 measured held MP4s gave 13,802 verified native
picture timestamps. Independent readback verified 1,024 proposed supporting
frames, and all decoded AAC waveforms aligned at zero samples. Four held
speakers supplied plain speech and six supplied Lombard speech.
[Development measurements](evaluation-panels/video-lip-signed-contrast-development-2026-11-18.json.gz) ·
[Held measurements](evaluation-panels/video-lip-signed-contrast-held-2026-11-18.json.gz)

## Replay

The standard-library reviewer checks every picture and treatment label
against its frozen roster, recomputes the development selection, and
recounts the held results. Native replay uses the same publisher archives,
local checkpoints and previously verified MP4 deliveries as the earlier
[signed timing study](video-lip-signed-rescue-2026-11-18.md#replay).

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
panels=docs/evaluation-panels
prior="$panels/video-lip-signed-rescue"
current="$panels/video-lip-signed-contrast"
before="$prior-plan-2026-11-18.json.gz"
plan="$current-plan-2026-11-18.json.gz"
development="$current-development-2026-11-18.json.gz"
held="$current-held-2026-11-18.json.gz"
python3 -B -S -m scripts.video_lip_signed_contrast review "$plan" "$before" "$development" "$held"

runner=backend/.venv/bin/python
old_cal="$prior-calibration-2026-11-18.json.gz"
old_held="$prior-held-2026-11-18.json.gz"
"$runner" -B -m scripts.video_lip_signed_contrast run \
  "$plan" "$before" "$old_cal" "$old_held" "$prior_media" "$sources" "$models" "$timing_assets" \
  "$work/development" --phase development
"$runner" -B -m scripts.video_lip_signed_contrast run \
  "$plan" "$before" "$old_cal" "$old_held" "$prior_media" "$sources" "$models" "$timing_assets" \
  "$work/held" --phase held --frozen "$work/development/receipt.json.gz"
```
