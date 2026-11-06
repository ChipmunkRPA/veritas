# Complete-video temporal learner: frozen development protocol

This experiment is research only. It does not change serving, confidence,
payment, retained results or infrastructure. Freeze this protocol before
generating fitting videos; freeze executable feature/learning code before
extracting new detector responses. No hyperparameter sweep or post-result
gate changes. The prior V8 research candidate remains the reference.

## Sources and construction

Keep the V6 source-role plan: sixteen fitting actors, four ordinary point
actors, eight consumed development actors, mutually disjoint. Use all sixteen
fitting sources and every exact common video/audio frame (51–71 pictures).
Trim original mono16k PCM to the picture origin and complete 25Hz spans;
never pad, repeat or synthesize sound. New lossless FFV1/PCM clips start at0.

Each fitting source supplies ten clips: original, local-light1.12,
local-resample, muse-resample, local-blur-2, local-blur-4, ai-redraw,
ai-retime, muse-ai-aligned and muse-ai-earlier. All edits occupy the centred
twenty source frames, with two-frame edge opacity and exact changed-pixel
masks. Use the already verified MIT LivePortrait human weights (restricted
face components excluded), MIT MuseTalk/VAE, Apache Whisper/face task and
permitted face geometry. Existing source and checkpoint hash receipts apply;
no new assets. Source-dataset terms remain separate from checkpoint licensing.

Run original-frame mesh sequentially on every retained picture. Portrait
redraw uses the same original aperture; retime uses eight source frames
earlier. Speech rendering binds original audio clock and uses the current or
eight-tick-earlier conditioning. The common CPU latent seed per source/frame
is SHA256(`chipmunk.mouth-temporal/11:{source_id}:{frame_index}`), first eight
bytes unsigned big-endian modulo2^63. These are construction labels, not proof
that a requested lip-sync was achieved. All160 clips must be prepared; retain
failures instead of silently substituting sources or conditions.

## Inference features

Both V8 and V9 learned vision heads remain frozen. Enumerate every eligible
V10 tracked window, with native three-frame centre means and two centres on
each flank. Use49 inference-available features in fixed order:

* For each head: middle mean, average flank mean, V10 relative response,
  absolute flank difference, middle standard deviation, relative response /
  (.1+middle standard deviation), middle standard deviation /
  (.1+combined-flank standard deviation): fourteen values.
* Mesh traces: inner-lip aperture/width (13,14 /78,308), mouth width/eye
  distance, nose1 projection on eye33→263 axis /eye distance, perpendicular
  projection /eye distance, unwrapped eye-axis roll, eye distance/native
  width. For each trace: middle mean, absolute middle-minus-flank mean,
  absolute before-minus-after mean, middle standard deviation:24 values.
* Native aligned mouth-chip grayscale mean/255, standard deviation/255,
  Laplacian variance/255². For each: middle mean, absolute difference from
  combined-flank mean, middle standard deviation: nine values.
* Window duration and number of middle centres: two values.

Mesh runs sequentially on the analyzed video itself, not on the corresponding
original. Bind source pixels, geometry and actual native measurement chips.
Require unique finite geometry for every used picture; missing measurements
make that clip unassessed, never original. No source identifier, original
reference, generator name, condition or truth mask enters the feature vector.

## Fitting, point and advancement

Use all complete fitting windows. A positive training window must belong to
an AI clip, have time IoU>=.2 with its known interval and its representative
centre must overlap an actual changed mask. Other windows are negatives.
Every fitting actor and all four AI conditions must supply a positive window.
Balance actor, class and condition; normalize sample weights to the number
of windows. Weighted fitting-only mean and scale (floor1e-3), logistic C=.1,
lbfgs tol1e-8, max_iter2000, random_state2718. Refuse non-convergence.
This49-variable linear classifier outputs a raw signed margin, not probability.

Choose its review point solely from the existing24 ordinary V10 point clips:
nextafter(max(0, all window margins), +infinity). Keep all four point actors
and all six ordinary conditions. Preserve V8 absolute evidence first; use the
temporal learner only when V8 has no flag. Evaluate all88 consumed development
clips with the same V9 advancement function: at least six net additional
locations; at least six LivePortrait locations including two each redraw and
retime; at least thirteen MuseTalk locations; lose at most one V8 success;
at most one additional ordinary flag total and per condition; mean spatial
IoU at least V8+.02; time IoU and assessment coverage retained; exact current
and retained-head native/response parity. Do not lower these gates after seeing
results. An unassessed clip is not a negative outcome or an original verdict.

Audit exact construction, every native input and geometry/feature arithmetic;
retain source receipts, all windows and fitting coefficients for independent
refitting. Passing this development panel still requires unseen actors AND
new sentences, multiple generators and realistic benign transformations,
outside review, customer evidence/export/payment integration and a separate
overall-confidence admission. GCP remains last.
