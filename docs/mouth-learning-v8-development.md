# Two-family mouth representation learning: frozen development design

This experiment updates a representation rather than relaxing a rejected cue.
It is consumed-source research; no customer detector, calibration or cloud
setting changes. The fixed v6 roles remain: 16 fitting actors, four separate
point-selection actors, and eight excluded actors used only for development.
All 28 actors have been consumed previously. No new actor is scored here.

## Matched fitting inputs

Preserve all 800 v6 PNGs and labels. Add the v7 permitted MuseTalk renderer on
the same eight selected frames of the same 20 original recordings. Its five
conditions are ordinary resampling, local Gaussian blur sigma 2 and 4 native
pixels, nominally aligned neural redraw and eight-tick-earlier neural redraw.
Reuse the frozen v7 YuNet landmark-box approximation, mouth ellipse, two-frame
edge opacity, strict asset loader and lossless PNG encoding. No publisher demo
footage or additional checkpoints. Source ODbL/DbCL terms remain separate from
the generator MIT and encoder Apache-2.0 declarations; these declarations do
not establish complete training rights or pretraining disjointness.

Record original source PTS and audio mapping. Select the conditioning index by
the picture's source time relative to the contiguous decoded audio origin,
requiring exact 25 Hz ticks. Do not assume that picture frame zero starts at
audio sample zero. Preserve the eight-tick offset as requested conditioning,
not proof of achieved lip synchronization. Use the same per-source/frame CPU
posterior seed for both neural conditions, derived from the first eight
big-endian SHA-256 bytes of `chipmunk.mouth-learning/8:{source_id}:{frame_index}`,
modulo 2**63. Retain exact masks, original/neural input bindings and transforms.
All 800 additions must be ready; no label/source substitutions or partial fit.
Together there are 1,600 stills (1,280 fit, 320 point), not independent people.

## Fixed learning method

Use the same permitted pinned DINOv2-small checkpoint as v6. Production YuNet
and GenD alignment select the native mouth/lower-face rectangle x=[.28,.72],
y=[.60,.92], floor/ceil bounds, then bicubic resize to 112 and ImageNet
normalization. A crop denotes the measurement support, not causal attribution.

Freeze embeddings and encoder blocks 0 through 9. Actually train blocks 10
and 11, final layer normalization, and a new 768-to-1 linear head. Concatenate
the separately L2-normalized final CLS and mean patch vector. Keep all layers
in evaluation mode (dropout/stochastic depth disabled) while gradients update
the declared trainable parameters. Float32 MPS, seed 1729, four CPU threads.
Initialize the head from CPU normal(0,.01), bias zero. Fixed 12 epochs, batch
32, AdamW (betas .9/.999, eps 1e-8, weight decay .01), learning rates 1e-4 for
the two blocks/norm and 5e-4 for the head. Constant learning rate; no early
stopping, checkpoint selection or architecture/optimizer search. Clip total
trainable gradient norm at 1.0. No autocast or silent device fallback.

Cache exact frozen-prefix tokens to reduce repeated computation. Fit inputs
have four fixed views: native, horizontal flip, JPEG quality 75, and flipped
JPEG75. JPEG uses OpenCV encoding/decoding on the native mouth chip before
resize/normalization. Within each epoch choose view `(epoch+row_index)%4`;
shuffle fitting rows with a CPU generator seeded 1729+epoch. Point-selection
and video development use native view only. Each fitting actor has equal
weight, each class half its actor weight, with equal shares for its conditions
and selected frames. Weights sum to the fitting row count. The per-batch loss
is mean(weight * binary cross entropy); no point actor enters any gradient,
normalization, augmentation selection or model selection. All 12 epochs run;
only the final state is evaluated. Keep token/view hashes, per-epoch loss,
parameter names/counts and frozen-weight identity checks. Verify cached-prefix
and ordinary full-model forward equivalence before fitting.

Select the raw review point just above max(0, all non-AI native margins from
the four point actors). Their AI margins do not determine it. Responses are
raw margins, not calibrated authorship confidence or public screening cutoffs.
Freeze implementation, training manifests, baseline receipts and this design
before any feature extraction/training/development responses.

## Two-family development decision

Use all 48 v4 plus all 40 v7 consumed clips, exactly as published. Reproduce
all unchanged baseline windows and native inputs using the corrected v5 crop
association, never sorted-cache positional assumptions. Use v6 triplet means,
window/context eligibility/ranking and native measurement-region projection.
No development maxima may change the point. Score all 88 planned outcomes,
including execution failures and unavailable inputs, and each family separately.

A first flagged window counts as located only at temporal IoU >= .2. Spatial
IoU uses the exact known mask at its representative source frame; a missing
mask/out-of-edit frame contributes zero. Advance to a fresh confirmation
design only if all hold: >=6 additional correct AI locations in total;
>=2 in each editing family; at most one lost baseline location; total ordinary
flags <= baseline+1 (currently 1+1), and <=baseline+1 per ordinary condition;
mean spatial IoU over all 32 edits improves by >=.02; no reduced assessed
coverage; all 88 executions and baseline/native input replays pass. These are
development gates, not accuracy claims. Failure preserves the running checks.
A pass still requires separately frozen unseen actors AND new sentences,
multiple generator families, compression/resolution and ordinary edits,
independent review, UI/PDF/JSON/paid-flow parity and its own calibration before
any overall-confidence use. GCP remains last. No outside review is implied.
