# Source-matched mouth learning: fixed v9 development protocol

V8 localized 14/16 MuseTalk edits and 0/16 LivePortrait edits, with no ordinary
flags on 56 controls. Its both-family advancement gate failed. Preserve that
result. V9 tests a distinct source-matched objective, not a lowered point or
an assertion that the failed V8 experiment passed. No serving/default,
confidence, billing or cloud changes during research.

## Immutable data, initialization and hypothesis

Reuse all 1,600 V8 native stills and all 5,440 exact prefix/view tensors, their
source/mask audits and unchanged 16 fit/four point/eight video actor roles.
No new media or model downloads. All subjects, sentences and point actors
have been consumed previously; none is fresh confirmation. Components and
source terms are those recorded in V6–V8 (permitted Apache-2.0 small backbone,
MIT generator components, CREMA-D ODbL/DbCL). Full training-rights/pretraining
disjointness remain unestablished. Do not discard difficult same-frame redraws.

Initialize the single learner from the exact retained V8 learned state/head.
Freeze embeddings and blocks 0–9; update only blocks 10/11, final norm and
the same 768-to-1 head. Keep a separate, entirely frozen V8 teacher during
fitting. There is only one candidate model at evaluation/inference, with no
generator label or clean reference available to it. The hypothesis is that
matched edited-versus-original/resampled pairs can teach subtle traces while
teacher retention preserves useful second-family behavior. This is not proof
that those traces are identifiable in every file.

## Predeclared fitting objective

Float32 MPS, four CPU threads, seed 2718. Evaluation mode throughout; no
dropout/autocast/silent device fallback. Sixteen fixed epochs, batches of 32,
AdamW betas .9/.999, eps1e-8, weight decay .01; constant learning rates 5e-5
for the two blocks/norm and 1e-4 for the head. Clip trainable gradient norm1.
No early stopping, checkpoint selection or hyperparameter search.

Each step uses the sum of these three terms:

1. Source/class/condition-weighted binary cross entropy on the shuffled 1,280
   fitting rows, using exactly V8 weighting and view `(epoch+global_row_index)%4`.
2. **0.5 × mean softplus(.75 - (positive_margin-negative_margin))** on a batch
   of 32 matched LivePortrait pairs. Each positive (`ai-redraw`, `ai-retime`)
   pairs with `original`, `local-light`, and `local-resample` at the exact same
   actor/source/frame, giving 768 unique fitting pairs. Require matching
   source hash, original RGB hash and role. Use the same view on each side,
   `(epoch+pair_index)%4`. Shuffle all pair indices using CPU seed3718+epoch
   and cycle that order for 40 fitting steps (1,280 pair draws per epoch,
   every unique pair used, 512 repeated). No point/development pair is eligible.
3. **0.25 × mean(weight × squared(student_margin-teacher_margin))** on the
   current BCE batch's non-LivePortrait-positive rows. This includes both
   MuseTalk AI conditions and all ordinary conditions; LivePortrait AI rows
   are excluded. Cache the frozen teacher's fitting-only view outputs (4,096
   offsets). Point actors cannot affect teacher targets or gradients. Verify
   all teacher parameter hashes remain unchanged.

Shuffle fitting rows with CPU seed2718+epoch. Retain per-epoch terms, exact
pair identities/offsets, teacher target hashes and changed/frozen parameter
identities. Use only the final epoch. Select a new raw review point just above
max(0, all ordinary native responses from the same four separate point actors).
Their AI responses cannot select it. This is internal raw-margin calibration,
not calibrated AI probability or a public screening-threshold claim.

## Frozen comparison and advancement

Before any teacher extraction/training/V9 development response, freeze code,
this protocol, V8 cache/head/weights/88-case receipts and all source audits.
Use the exact 88 V4/V7 clips, native triplets, corrected crop association,
V6 window/ranking/measurement-region rules. Preserve current production
baseline parity on all clips and verify every V9 native input against the
retained V8 binding, except learned-feature hashes/margins that should change.
Report current cue, V8 and V9 separately. Missing/failed cases stay in all
denominators; absent flags do not establish an original file.

Advance only if all hold against immutable V8: >=6 net additional correctly
located AI edits; >=6/16 LivePortrait locations, including >=2 redraw and >=2
retime; >=13/16 MuseTalk locations; <=1 lost previously correct location;
total ordinary flags <=V8+1 and <=V8+1 per ordinary condition; mean spatial
IoU over all32 AI edits >=V8+.02; mean temporal IoU >=V8; no lost assessed
coverage; all88 current-baseline and V8 native-input replays pass. A location
requires first flagged-window temporal IoU>=.2; out-of-edit representative
frames/missing masks contribute zero spatial overlap.

Failure is retained and rejected without lowering the point or spending fresh
actors. A pass permits only a separately frozen fresh confirmation, with unseen
actors AND new sentences, multiple generation families, realistic compression/
resolution/ordinary controls, outside review and customer evidence/paid-flow
integration. Overall-confidence admission remains separate. GCP remains last.
