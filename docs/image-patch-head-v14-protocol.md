# Supervised image patch head: development protocol

The current JPEG region selector remains weak, and the v12 recalibration and
v13 handcrafted texture candidates were rejected. This new experiment learns a
regional head on a frozen pretrained visual representation. It is independently
implemented and is not a reproduction of EISL, TRAIL or a claimed new benchmark.
It does not change serving models, public conclusions, billing or confidence.

## Native representation and learned head

Use the already cached Apache-2.0 DINOv2 register-base model and reviewed local
architecture archive pinned in the model manifest. The exact weight SHA-256 is
`73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71`.
Verify both cached assets and commercial permission before construction; make
no download. The backbone contains 86,583,552 frozen parameters. Source weights
and code retain their Apache notice; independently written project code retains
the repository license.

Warp stored RGB to 448x448 using Pillow bilinear resizing and ImageNet channel
normalization. Extract normalized patch tokens from blocks 4, 8 and 12, excluding
the class token and four register tokens. Concatenate into a 32x32x2304 float32
representation. The original coordinate transform spans the complete source;
it is not a center crop or a claim about full-resolution pixel authorship.

Concatenate each token with its difference from the complete image's mean
feature. Fit a 4608->64 ReLU->1 sigmoid head, 295,041 trainable parameters, using
only fit-role cells. The backbone receives no gradient. The sigmoid is a ranking
response, not calibrated AI confidence. Attention gives each token global
context, so a regional response is an association rather than pixel-origin proof.

Use CPU float32, two Torch threads, deterministic algorithms and seed 20261108.
AdamW uses learning rate .001 and weight decay .01 for all parameters, exactly
30 epochs, deterministic shuffled batches of 512 and balanced soft positive/
negative supervision mass. No validation early stopping, checkpoint selection,
hyperparameter search or diagnostic-based arm selection is permitted.

## Source roles, supervision and controls

Reuse all 288 consumed source groups from v13: 120 fit, 60 point selection and
108 diagnostics, with all original/edit pairs and variants in their source role.
This is a development comparison, not fresh confirmation or an unseen-generator
test. Known model-pretraining overlap remains unestablished. The TGIF/TGIF2
publisher's CC BY-SA 4.0 dataset declaration and underlying photo rights remain;
retain photos, pairings, sampled features and fitted weights privately.

Use all 1,344 v13 case conditions without dropping sources: 840 fit inputs,
180 point-role inputs and 324 diagnostic inputs. Fit includes original quality-75
JPEGs, fixed lighting and JPEG55 interventions on originals/edits, plus the same
ordinary blur/tone patch on genuine images. Those interventions keep their
original source truth; ordinary edits and recompression receive no AI labels.

Pool the exact original RGB-pair edit mask into edge-aligned 32x32 cells using
actual source pixel counts. Pixel (x,y) maps to floor(32*x/width),
floor(32*y/height). Sample 64 deterministic cells per fit case, 32 with positive
edit area and 32 with zero edit area where both exist, with replacement only
for a class smaller than 32. Otherwise sample 64 distinct cells. The same source
cell indices accompany its interventions. Preserve partial-area targets as
fractions. Every source contributes the same seven fit variants in total.

Fit all 840 cases before scoring the genuine point role. Then score both
genuine original/ordinary variants of all 60 point groups. Select the descending
group-maximum value at index floor(.05*n), strict greater-than. Freeze the fitted
weights, fit-array hashes and point before scoring point-role AI files or any
diagnostic case. This is an internal research comparison point, not a public
screening threshold, user guarantee or authorship probability.

Median-filter the ranking grid with a 3x3 reflect boundary, retain 8-connected
strictly superlevel components of at least four cells, and project their exact
cell rectangles to source pixels. Four of 1024 cells matches the area floor of
v13's 16 of 4096 cells; it does not grant pixel-level resolution.

## Required comparison and continuation

Compare identical diagnostic originals/edits to their input-bound retained
current-selector records from v11/v12. This is a component comparison, not a new
full-application baseline neural run. Report all 108 edited, 108 genuine and
108 ordinary controls, per generator and prior-source cohort. Include masks
that miss the edit, false highlights, mean IoU/pixel F1 including empty masks,
and counts with IoU at least .3. Zero highlights never establish originality.

Continuation requires higher mean edited IoU, more IoU>=.3 cases, no extra
genuine flags versus current, and ordinary-control flags at most 5%, plus the
complete audit below. Preserve all rule failures and subgroup losses; do not
promote a successful subgroup. A passing result still needs actual current-check
ordinary-control inference, a separately frozen fresh confirmation and full-app
evidence/export verification. No serving admission follows from this study alone.

## Native qualification, audit and resources

Before project-source scoring, four self-authored procedural inputs must match
full-forward layer hooks exactly, and one native repeat must be exact: nine
backbone forwards. Publish code, protocol and a prospective source-panel freeze,
and verify the remote bytes before the first project-source forward.

The main run has 1,344 backbone case forwards, 18 prospective full-hook probes
(first case of each role/generator/class cell) and one exact first-source repeat:
1,363 backbone forwards. Head fitting has 3,150 updates, separately counted from
504 full response-grid inferences. Retain all native feature/tensor/source
commitments, lossless float32 fit samples and means, full response maps, point
snapshot and append-only journal. Full feature maps are committed rather than
duplicated; the separate audit must actually recompute them all.

The auditor repeats all 1,344 native feature cases plus 18 full-hook probes,
verifies every stored training sample and target, independently pools source
truth with integral sums, and verifies all 504 native ranking grids exactly.
A separate NumPy float64 dense/ReLU/sigmoid/median implementation must match
within 3e-5, with independent global means within 1e-5. Record cells near the point;
the exact retained/native grids, not rounded reference scores, govern the fixed
comparison. Independently traverse components, project cell rectangles, count
source pixels and refit the genuine-only point. The optimizer is not independently
refitted and no outside human review is claimed.

Each phase runs alone with a 900-second guard, at most 6 GiB sampled process-tree
RSS and at least 1,536 MiB available RAM/free disk. Sampling is a lower bound,
not an exact peak guarantee; retain sampling errors, exits and guard failures.
No implicit restart or omitted case after an interruption. Public artifacts
contain source/code/model commitments, methods and aggregate results only.
No customer files, external verification requests or GCP resources are involved.
