# Supervised regional image adaptation, v26

This is an actual learning study, following the rejected v18 generic localizer
and successful v24 full-resolution checkpointed differentiation test. It is not
a released detector or a detection-accuracy claim. No source inference occurs
before publication and remote verification of the preparation freeze.

## Sources and separation

Use all 397 previously qualified SD2 splice/source pairs, observed changed-pixel
masks, and non-generative tone and blur controls. The exact private v21 source
plan, qualified rows and owner-enriched v25 connected components are committed
by SHA-256 in the public preparation record. No new media is acquired.

Keep each complete creator/near-duplicate component in one role. Rank component
hashes with the fixed `chipmunk-regional-v26-role` salt. First reserve 48 of the
70 eligible components containing publisher validation/testing photos with no
identified prior-consumed link. From the remaining unlinked components reserve
32 calibration, 24 point-estimation and 48 diagnostic groups in that order.
All remaining groups, including all 12 components linked to consumed material,
are fitting data. These rules are fixed before inspecting new model responses.
Preserve every photo; no image, mask, format or small edit may be removed after
seeing a difficult outcome. Photos in a component are correlated observations.

The 14 unresolved legacy owners, unobserved aliases/bursts and the upstream
checkpoint's pretraining-photo overlap remain limitations. Source grouping
does not establish different human photographers or camera authenticity. This
single-generator study cannot demonstrate unseen-generator generalization.

## Learning method

Use the pinned RelayFormer image-only Apache-2.0 checkpoint and MIT source
already acquired and inspected in v18. The v20 photo license audit and v21
qualified-pair acquisition govern these local development sources. Their
underlying provenance limitations remain; no broader rights clearance or new
weight-redistribution claim is made.

Adapt exactly the 246 used local/global LoRA and mask-decoder parameter tensors
selected by frozen v23 code. Dense backbone, unused paths and positional
frequencies stay frozen. This learns backbone adaptations as well as the
decoder; it is not another frozen-feature linear readout. Full source geometry,
1024 canvas, 528-pixel overlapping patches and native reconstruction remain.
Use v24 non-reentrant block checkpointing, CPU float32, two threads, deterministic
algorithms, seed 20261108, and eval mode to disable dropout while differentiating.

Train for exactly six epochs over every fitting photo. Each epoch's order is
the hash of the fixed order salt, epoch and source ID. Even epochs use native
pixels and odd epochs JPEG quality 75 with Pillow defaults. Negative controls
are original, tone and blur respectively for epoch pairs 0–1, 2–3 and 4–5.
Every photo thus appears in all six format/negative combinations.

Each optimizer update accumulates two equally weighted losses: one AI splice
with its source-bound changed-pixel target and one corresponding negative with
an all-zero AI target. Retain truth coordinates before JPEG encoding. Reproduce
all input, truth and ordinary-control hashes before use. The frozen v23 loss
uses equal positive/background BCE plus half a soft Dice term for positives,
and mean BCE for negatives. The 397 audited source views have both foreground
and background and do not require the excluded all-positive loss branch.
Use AdamW, constant learning rate 0.0001, weight decay 0.01 and gradient norm
clipping at 1. No augmentation search, crop removal, best-epoch selection,
held-out early stopping or learning-rate tuning is allowed.

## Execution and preservation

Run explicit sequential phases of 32 paired updates, except the final shorter
phase. Each phase starts from the exact previous learned parameters and AdamW
state, bound by checkpoint, plan and update-journal hashes. A failed resource
profile cannot be resumed as if successful. Existing phase directories or
checkpoints are never overwritten. Preserve incomplete writes and logs.

Checkpoints remain private on the owner's connected Seagate drive. Active
serving assets stay local. Each phase records actual forward/backward/update
completion events and source/tensor/target geometry bindings. Check every
selected gradient is finite, frozen parameters receive no gradients, and dense
and fixed-frequency tensors remain bit-identical. Store the learned-parameter
and optimizer state plus hashes; publish only code, aggregate counts and hashes.

Observe process-tree RSS every 0.25 seconds. Limit each phase to 900 seconds,
6 GiB sampled tree RSS and at least 1.5 GiB RAM/internal/checkpoint disk reserves.
Require 6 GiB available RAM and 3 GiB free disks before starting. Every stopping
measurement is retained; sampled RSS is not a true peak guarantee. No automatic
restart, parallel heavy phase or unrecorded fallback is permitted.

## Evaluation and release boundary

Use only the final six-epoch checkpoint for subsequent calibration/evaluation.
Training loss does not establish improved detection. Before any inference on
the reserved sources, separately freeze readout, calibration, baseline comparison,
false-alarm matching, uncertainty calculations and the confirmation decision.
The reserved sources are not used for fitting or choosing an epoch. Raw
localization responses are not probabilities of pixel authorship.

Adoption still requires actual current native/JPEG checks on the same reserved
source-bound originals, retouches and AI splices; useful localization, missed
edits, false alarms and inconclusive/failed denominators; and broader fresh
commercially permitted generators. No serving/confidence or protected audio
calibration changes follow from completing training. No outside reviewer has
reviewed this study. No outreach, GitHub workflows or GCP work is authorized.
