# Full-app frame check on a genuine masked speaker

Use the verified USGS group 03 CRF24 case for both arms. The publisher marks
the employee-interview compilation public domain and lists no separate
soundtrack restriction. This test uses the unmodified speech and a normal
recompression of the genuine picture. The single visible lead speaker
wears a mask. The recording comes from a different publisher and event
than the Nairobi case. Shared content with other historical panels and
pretraining overlap have not been comprehensively checked. Retain the
source attribution and keep the clips and derived exhibits local.
[Official source and credits](https://www.usgs.gov/media/videos/working-usgs-md-de-dc-water-science-center) · [HD admission and preparation](video-genuine-hd-source-admission.md)

The only scored file is the existing group 03 case 02. Its QP0 peer
verifies the common preparation clock and PCM. The native source excerpt is
`[1057056, 1237236)` on the 1/30000 picture clock. After rebasing, the
prepared clips have 180 native 1920 × 1080 pictures at 1,001-tick intervals.
The video ends at tick 180,180. Both prepared soundtracks contain 288,288
samples per channel at 48 kHz with zero audio origin. The constructor
records the original source offset of 1/120000 second and its removal
in rebasing.

The new freeze binds the v166 preparation, exact CRF24 input, independently
decoded RGB hashes for all 180 pictures, and the app-format 96,096 mono
16 kHz PCM samples. Both v171 arm manifests are copied byte-for-byte. Reuse
the corrected v171 environment, ordinary full-profile pipeline, separate
owned work directories, pinned assets and unchanged monitor. Network and
billing remain disabled. Each arm must verify its actual input clock and
sampled RGB, audio mapping, committed result and evidence assets.
[Prior full-app method](video-frame-full-app-v171-method.md)

Record the actual target status and reason, sampled frames, frames with
detected faces, scored frames, linked signal and frame-series coverage,
resource observations, saved validation slice, and customer assessment.
Check that the other detector statuses, fuser identity, and input
condition profile agree. A shared face-input hash also checks the sampled
positions and detection geometry across the two arms. Zero detected
faces are a valid coverage result
only when the face detector succeeded and both target routes report no
detected face with zero scores and no linked successful signal or series.
With detected faces, the existing reference budget skip is recorded
separately. Candidate admission requires all detected-face frames to
have bound scores and a matching published series. A mask does not
establish either outcome before execution.

If a face interval is published, inspect its saved before/during/after
anchors against the committed native PTS and RGB. Zero intervals can be
appropriate on this genuine clip. This test provides one additional
control for coverage and possible false alarms. Face-swap sensitivity,
an accuracy rate, and readiness to change the default require a wider
rights-reviewed comparison.

Run from the repository root. The preflight checks the preparation and
decoders without opening a model lease. Freeze into a new exclusive
directory and run the two guarded arms sequentially.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_frame_full_app_v175 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_frame_full_app_v175 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_frame_full_app_v175 run reference
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_frame_full_app_v175 run batch-one
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_frame_full_app_v175 readout
```
