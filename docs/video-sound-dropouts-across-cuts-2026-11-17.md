# Sound dropouts while a mouth keeps moving across a video with cuts

Chipmunk Verify checks for near-silent soundtrack intervals while a visible
mouth keeps moving. The Full profile can now perform this check separately
inside each detected shot. The ordinary view lists the covered and unassessed
shot ranges. A qualifying shot links to its exact source time, three original
frames and mouth crops, a waveform, and a playable excerpt. The PDF and JSON
preserve the same observations and positions.

The check requires a continuous, zero-start picture and mono soundtrack clock
in a video up to 15 seconds long and 360 frames. Each of two to five detected
shots is evaluated separately when it has 34–120 pictures and enough visible
mouth observations. The existing 5 ms amplitude measurements and motion
criteria apply. The quiet interval, the visible motion and the audible flanks
must fit within the same shot. Eligible source audio is 16 kHz PCM or 48 kHz
AAC. Muting, quiet movement and recording dropouts can produce this finding.
It contributes zero weight to the AI assessment.
[Single-shot measurement and supported assets](video-silent-mouth-2026-11-16.md)

## Local confirmation

The panel uses 18 new Lombard Grid target recordings from nine speakers absent
from the comparison's development speakers. It assembles two talking shots and
a 0.720-second title card. The licensed original soundtrack stays intact
outside each known change. Tests compare 240–520 ms hard and faded mutes inside
the later shot, quieting without a mute, unchanged sound and mutes across the
cut. The native readback verifies the exact picture stream, all shot boundaries
and the presented audio extent. A useful finding must reach 0.40 temporal
intersection over union with the staged interval. The paired encodings and
treatments share the same sources. [Lombard Grid and CC BY 4.0 terms](https://zenodo.org/records/3736465)

| Confirmation outcome | PCM | AAC |
| --- | ---: | ---: |
| Hard mute correctly located | 12/18 | 12/18 |
| Faded mute correctly located | 12/18 | 12/18 |
| Unchanged, attenuated or cross-cut controls flagged | 0/54 | 0/54 |
| Misplaced findings | 0 | 0 |

The earlier near-silence route skipped all these clips under its single-shot
scope. The new route assessed every delivery and marks the short title card
unassessed. The counts describe constructed camera cuts with selected visible
talkers. The qualifying rule retains its limits on the near-silent amplitude,
visible mouth motion and adjacent sound.
[Numeric receipts and exact input commitments](evaluation-panels/video-silent-mouth-shots-held-2026-11-17.json.gz)

A second rendering cuts directly between speakers. The cut detector supplied
the intended boundary in 10 of 18 videos. The new route located 14 of 20
dropouts and flagged none of 30 controls in each encoding on those 10
assessed videos. The other eight lacked a detected cut and remain outside
this comparison. The direct and title-card renderings reuse the same clips.
[Direct-cut dropout results](evaluation-panels/video-silent-mouth-direct-held-2026-11-17.json.gz)

## Separate test of replaced spoken words

A separate candidate compared mouth motion with speech independently inside
each shot. A calibration from 24 development montage groups set two operating
points from the nearest-rank 90th percentile of the ordinary montage maxima by
encoding. The earlier single-shot settings served as a fixed second comparison.
The fitted montage points recovered four of 24 development replacements in
each encoding and flagged two of 24 unchanged controls in each.
[Frame-level speech and mouth research](https://arxiv.org/html/2610.30872v1)
· [Development measurements](evaluation-panels/video-lip-multishot-development-2026-11-17.json.gz)
· [Frozen comparison](evaluation-panels/video-lip-multishot-freeze-2026-11-17.json.gz)

The frozen montage points missed all 18 new word replacements in each
encoding. The earlier single-shot points caught one per encoding, misplaced
three ranges and flagged two of 18 unchanged controls and two of 18 gain
controls. A direct-cut comparison recognized the intended boundary in 10 of
18 pairs. At the montage points it located zero edits and misplaced one
finding per encoding in those assessed clips. The word-replacement candidate
stays in evaluation.
[Source plan with excluded recording hashes](evaluation-panels/video-lip-multishot-held-sources-2026-11-17.json.gz)
· [Title-card results](evaluation-panels/video-lip-multishot-held-2026-11-17.json.gz)
· [Direct-cut results](evaluation-panels/video-lip-multishot-direct-2026-11-17.json.gz)

## App replay

Five simulated USD 9.99 checkouts created no analysis jobs before payment.
The browser, PDF and JSON showed the PCM and AAC evidence on the same
source ranges and frames as the staged deliveries. The report linked the
findings from the coverage list. Signed source replay confirmed the cut
boundaries, original frames, mouth crops and waveform. The matched
ordinary clip in each encoding showed no finding. The paired switch-off
run returned the same overall assessment.
[Local browser, checkout and export receipt](evaluation-panels/video-silent-mouth-shots-app-2026-11-17.json)

## Reproduce the comparisons

Use the existing local video and near-silence settings, including
`VERITAS_VIDEO_LIP_SILENT_MOUTH_REVIEW=true`. The media stay in the local test
directory. The receipts and published recipe contain no customer files.
Obtain the pinned archive files and local assets using the existing source
preparation instructions. [Publisher and asset preparation](video-neural-phone-clock-2026-11-14.md#independent-replay)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
panels=docs/evaluation-panels
roster="$panels/video-lip-multishot-held-sources-2026-11-17.json.gz"
python3 -B -S -m scripts.video_silent_mouth_shots_panel review "$panels/video-silent-mouth-shots-held-2026-11-17.json.gz"
python3 -B -S -m scripts.video_lip_multishot_calibration compare \
  "$panels/video-lip-multishot-held-2026-11-17.json.gz" "$panels/video-lip-multishot-freeze-2026-11-17.json.gz"
"$runner" -B -m scripts.video_lip_multishot_panel materialize "$sources" "$roster" "$run/native"
"$runner" -B -m scripts.video_lip_multishot_panel compare "$run/native" "$roster" "$assets" "$run/montages"
"$runner" -B -m scripts.video_silent_mouth_shots_panel measure "$run/montages" "$assets/video-lip-timing" "$run/silence"
npm --prefix frontend run build
"$runner" -B -m scripts.video_lip_multishot_app "$run/silence" "$assets/video-lip-timing" "$run/app" --browser
```
