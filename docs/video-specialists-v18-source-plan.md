# Video specialist calibration: source audit and remaining work

The first source audit for a four-role video specialist comparison is complete.
It loaded no detector and fitted no calibration. All 162 source-video groups
were previously consumed in development; the future reserved role is not a
fresh external benchmark. Existing local video behavior remains unchanged.

## What the audit actually verified

The audit includes every paired source group from six pinned RTFS-10K acquisition
manifests. Each group has an original target and one output from each of two
face-swap generators. It recomputed original, matched and retained interval file
hashes against their acquisition receipts. It joined known source-video URLs,
donor filenames and exact raw/derived byte identities before assigning roles.
The resulting graph has 162 known components, each containing one source-video
group. A salted component hash, followed by round-robin assignment, supplies
41, 41, 40 and 40 groups to specialist calibration, combination fitting,
combination calibration and reserved comparison. Detector responses do not
choose these roles.

There are 432 retained interval clips from 114 groups: 228 inserted-swap clips,
114 unmodified clips and 90 ordinary-edit controls. Positive clips have retained
exact-frame preparation receipts. The earlier source preparation verified frame
content; a new inference adapter must independently check actual decoding,
timestamps and local labels against these bindings.

A separate check of all 432 retained production probe receipts found an important
nominal-clock mismatch: 31 ordinary jump-cut controls are actually 25 fps, while
the old preparation manifest calls them 24 fps. Substituting nominal frame rate
would shift a full-clip duration by as much as 235 ms. All positive intervals and
original controls use measured 24 fps; the other ordinary controls use 24 fps.
The [retained-clock receipt](evaluation-panels/video-specialists-v18-retained-clock-audit.json)
records these counts and exact journal commitments. It reads previously measured
native timestamps; it does not claim 432 new decodes or detector forwards.

The new input adapter uses actual PTS for every file, retaining the measured
clock for older ordinary controls. New ordinary controls explicitly request
24 fps. A tiny authored native encoding regression verified that a 120-frame
input produces the intended 112-frame jump cut at 24 fps, with exact retained
frame content. It is a software regression, not fresh detection evidence.

The [aggregate receipt](evaluation-panels/video-specialists-v18-source-audit.json)
records actual path/byte counts, role coverage and source-manifest commitments.
The private inventory retains source identities. No raw clip, donor image,
embedding or source identity list is in this public package.

## What still needs development and testing

The existing interval coverage leaves 48 groups needing source-bound splice
preparation. There are also 72 groups without an ordinary-edit control. Include
these planned denominators before inference; do not select replacements from
their eventual detection scores. Keep original/AI variants and ordinary controls
in their connected source role. New preparation must verify known edit boundaries
and source-frame correspondence without storing unnecessary duplicate media.
The input adapter is implemented and its clock/truth tests pass. Its full
648-clip real preparation is queued after the current audio pipeline; that
preparation has not yet run. Temporary group renderings are deleted only after
their byte/clock commitments are written. Original research clips remain intact.

The first detector comparison will separate face-crop authenticity evidence from
within-track appearance and photometric changes. A local window needs its own
overlap label; a whole-file swap label cannot be copied into an unchanged window.
A complete clip summary and its suspicious intervals have different targets and
need separate calibration. Dependence between visual readouts must be accounted
for before fitting and calibrating a file combination. Skipped or unsupported
checks remain explicitly unassessed. The shared support floor stays at 25 known
source groups per class; more frames or derived clips cannot inflate that count.

The scoring policy, native input audit, resource limits and improvement gates
still need to be frozen before new detector inference. The source-role inventory
alone is not a completed experiment protocol. Run heavy video work after the
current sequential audio study, so both evaluations keep meaningful resource
measurements. Compare missed swaps, false flags on genuine/ordinary edits,
coverage and localization before considering a later fresh-source confirmation.

This is a face-swap comparison. It cannot by itself calibrate generated speech,
lip-generated video or general video synthesis. Those targets need their own
commercially usable checkpoints, positive/control sources and source-disjoint
validation. Ordinary audio offsets and mouth motion are not AI truth.

## Licensing and independence limits

The [pinned RTFS-10K publisher card](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)
declares CC-BY-SA-4.0. This is a publisher declaration, not an independent guarantee
of rights over every underlying recording or face donor. Exact source/donor-file
separation does not prove that all depicted people differ. External detector
pretraining overlap also remains unknown.

Existing pinned YuNet and SFace assets have recorded MIT and Apache-2.0 grants.
The [SFace upstream directory](https://github.com/opencv/opencv_zoo/blob/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/README.md)
explicitly covers its files, including its ONNX assets. A permissively licensed
repository alone is insufficient clearance for a checkpoint hosted elsewhere;
additional mouth/lip detectors need weight-specific primary-source review. No
new mouth checkpoint was acquired or admitted in this source audit.

The public source audit is reproducible with the original private research
cache. Five provenance regression tests verify transitive donor/source/byte
linking, historical-split leakage prevention and missing-donor handling. No
external human review, fresh-media accuracy, customer probability or court-proof
claim is made.
