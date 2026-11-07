# Image and region calibration under compression

This research pilot compares separate whole-image and regional response mappings
and their combination. All 240 source pairs have already been consumed in older
development studies; the four new source-disjoint roles do not make them a fresh
external confirmation set. No serving confidence or model default may change from
this pilot alone.

Use all 240 original/edit pairs from the four declared TGIF/TGIF2 manifests,
80 sources in each of Firefly, SD2 and FLUX Fill. The publisher distributes the
[dataset under CC BY-SA 4.0](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea).
Retain raw images and reference masks privately. Verify every original byte hash,
geometry and exact RGB edit mask against its genuine source before model scoring.
Within each family, hash-order source IDs and assign them round-robin to four
roles, 60 source groups per role. Refuse exact original/input byte collisions
across roles. External pretrained-source overlap remains unknown.

Use the already installed, pinned Apache 2.0 TIDAL weight and unchanged CPU
float32 loader, at two threads. Acquire no new model, execute no remote code,
and retain the current production JPEG check. The TIDAL JPEG arm here is
experimental; it does not replace the serving JPEG detector.

For every original/edit image, process its original PNG and deterministic Pillow
JPEG quality 75 bytes. Bind each derived input's file and decoded RGB hashes
before scoring. Compression is a measured condition and an alternative
explanation, not an additional AI edit. Preserve original edit-mask coordinates
for the JPEG pair. Do not label ordinary recompression as synthetic pixels.

Use a whole-image 95th-percentile response and four local means from deterministic
quadrants that cover every stored-source pixel exactly once, including odd image
dimensions. Project the native 384-square float map to the original geometry with
nearest-neighbor interpolation. Each quadrant's label comes from its own actual
reference-mask intersection. A positive image can have genuine quadrants. These
are spatial associations under a global receptive field, not proof of pixel origin.

Freeze four arms: whole-image pooled calibration; whole-image separate PNG/JPEG
calibration; whole plus regional pooled calibration; whole plus regional separate
PNG/JPEG calibration. All readouts share one dependence family. Pool their maximum
positive calibrated logit, then independently fit and calibrate the file response.
Use the shared 25-source-group/class support floor, L2=.001 and 50:50 reference
prior. Bind the complete confirmation byte/geometry/local-truth plan before any
fitting; fit every arm before any confirmation forward. Insufficient support or
missing evidence cannot become an originality estimate.

Repeat the first original/edit input per role and format bitwise. Compare every
available native PNG output to the original 144-map retained journal. Retain
compact source/response hashes and scalar readouts, rather than duplicating
hundreds of full response arrays. Bound each run to 30 minutes, 6 GiB sampled
process-tree memory and at least 1.5 GiB available memory/free disk. Preserve
failed attempts, partial journals and native differences; do not silently restart.

Report source-balanced file Brier loss, false AI calls, missed AI and all
unassessed/tied cases, including PNG/JPEG denominators. An internal 0.5 tendency
readout is a descriptive research comparison under the stated reference prior,
not a public screening threshold. This panel does not establish full-image
generation, face-specific detection, independent outside review, customer
correctness or suitability as court evidence. Video and noisy-voice calibration
remain separate required work under the four-media plan; GCP remains last.
