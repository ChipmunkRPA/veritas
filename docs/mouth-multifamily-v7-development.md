# Second-family mouth edits: consumed-source extension v7

This expands the known-edit benchmark; it does not admit a detector or alter
customer confidence. Use all eight already-consumed v4 original clips, in their
existing source order. No new actors or source sentences are selected and no
fresh confirmation is consumed. Source media and checkpoint binaries stay out
of Git. Public source metadata retains CREMA-D's separate ODbL/DbCL attribution.

## Generator and permissions

Use the pinned permitted asset plan in `mouth-v7-generator-assets.json`:
MuseTalk V1.5 human weights (publisher MIT/commercial declaration), the MIT
SD-VAE and Apache-2.0 HF Whisper-tiny. Reuse permitted YuNet geometry. Exclude
upstream internet demonstration media, DWPose, S3FD, face-parsing checkpoints
and optional SyncNet. No downloaded upstream Python/shell file is executed.
Publisher permissions are not a complete training-data-rights audit.

The Mac port uses float32, explicit MPS, strict checked tensor loading, a
lossless 16-key VAE attention-name conversion and the speech encoder only.
VAE posterior samples use a seeded CPU generator, copied to the selected
device. Conditioning uses ten 50 Hz ticks/all five encoder states per 25 Hz
picture frame, with explicit edge zeros. It supports at most 30 seconds here.
It is adapted inference, not a claim of stock-pipeline or cross-device parity.

The successful fixed-frame smoke required exact repeated-seed pixels, changed
pixels under changed audio conditioning, and no composite change outside its
mouth support. It found a closer permitted crop approximation before any
detector response: use a YuNet face box with vertical half-face position
eye_y + 0.75*(nose_y-eye_y), reflected to the detected chin. This approximates
the publisher's 68-point landmark-29 box rule; it does not measure that landmark.
The native rectangular crop is resized to 256 with Lanczos4 and retains an exact
pixel-edge mapping. The earlier aligned-crop smoke is retained separately.

## Fixed construction

The existing lossless original clip, its native frames, exact PTS and unchanged
mono 16 kHz PCM are the parent input. Verify its bytes against the v4 manifest.
Retain its full 63–100 complete frames; do not loop or pad speech/picture.
Every source gets the same centred 20-frame edit interval (0.8 s) used in v4.
For each edited frame require exactly one eligible face; any source failure
remains an unavailable planned case, without replacement.

Prepare five new variants per source (40 planned clips):

1. Ordinary crop/resampling using exactly the model's permitted source crop.
2. Ordinary local Gaussian blur, sigma 2 native pixels.
3. Ordinary local Gaussian blur, sigma 4 native pixels.
4. Neural redraw using that source frame's nominal aligned audio conditioning.
5. Neural redraw conditioned on audio eight picture frames earlier.

Blur controls are included because the engineering render can soften local
detail; softness alone must not count as evidence of neural editing. The three
ordinary conditions have label zero despite known non-AI editing. Both neural
conditions have known AI provenance. Requested aligned/earlier conditioning is
not proof of achieved lip-sync quality or an exact measured 320 ms offset.

Each variant uses the same feathered native mouth ellipse from YuNet mouth
corners, axes 0.7 and 0.45 times corner distance, and two-frame temporal edge
opacity. Each source/frame's neural variants share the same fixed seed; only
conditioning changes. The untouched region must be bitwise equal. Retain exact
changed-pixel masks, source crop transforms, conditioning/model-input hashes,
renderer proof and before/during/after source ranges. Zero-change planned edit
frames are preparation failures, not convenient substitutions.

All clips share the v4 lossless FFV1/native RGB/PCM container recipe. Decode and
check every pixel, timestamp and soundtrack against the planned arrays. A
separate source replay must verify masks, surrounding frames and unchanged PCM.
Generation implementation, asset plan, parent manifest and this protocol are
frozen before preparing the panel. Preparation inspects no detector response.

## Current-check characterization

Freeze existing review code/point and complete outcomes before scoring the new
clips. Use the production CPU face-track/sequence path with verified commercial
YuNet/DINOv2-small weights, same source sampling and existing inspection point.
Report known edit interval overlap, native representative-region overlap,
ordinary-condition review flags, assessed coverage and all failures. Unassessed
and unflagged AI clips stay in the missed-localization denominator. Timing
measurements are context; synchronization is not an original-file verdict.

This is a baseline characterization on consumed actors with a new editing
family. Do not refit points or revive rejected v4/v5/v6 candidates using this
panel. A future learned method needs separate actor/source fitting and point
roles, then a new frozen confirmation on unseen actors AND sentences with
multiple editing families, ordinary edits and resolution/compression controls.
No outside review, UI/report integration or global-confidence calibration is
implied by generating or scoring this panel.
