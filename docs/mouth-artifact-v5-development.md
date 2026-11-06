# Learned face-artifact response: mouth-edit development comparison v5

This is a new, isolated comparison on the 48 consumed v4 mouth-edit cases from
eight CREMA-D actors. It tests a learned forgery response instead of generic
feature change. It does not train a new head, certify lip synchronization,
produce calibrated AI confidence, or alter serving behavior. A successful
development screen still needs a separately frozen fresh-actor/sentence test.

Revision 5.1 repairs an experimental input-binding error, without changing the
statistic, cases or advancement gates. The initial v5.0 runner wrongly paired
capture order with sorted feature indices; FrameStore returns cached frames
first. All 48 independent native-pixel audits failed, invalidating that run's
localization figures. Its code is retained at Git revision `549c89e` and its
local receipts remain immutable. Revision 5.1 associates native crops with the
adapter's original insertion keys before feature sorting and verifies the exact
112-pixel production crop at that boundary. Every frame must pass the separate
source/tensor audit before interpreting the new comparison.

Use the already installed GenD CLIP L/14 checkpoint at registry revision
`891ce014a0308386c4d7d25b3dcf436a22db5504`, weight SHA-256
`d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833`.
Its publisher declares MIT, as does the base CLIP encoder. Replace the upstream
restricted face detector with the already permitted production YuNet weights.
Publisher permission is not a complete training-data-rights audit; pretraining
overlap with the source actors is unknown. No new checkpoint is downloaded.

The production face-sequence adapter must run unchanged on every case. Capture
the native-resolution aligned crops before its 112-pixel DINO resize, and its
actual tracked input arguments. Verify its response, windows and sampled centres
against the immutable v4 receipts, not a recreated approximate baseline. The
baseline remains CPU; the candidate uses the existing float32 GenD implementation
with batch size four on MPS. Retain hashes of native RGB, alignment coefficients,
native crops and preprocessed float32 model tensors. No ground-truth masks or
edit boundaries may select inputs, crops, windows or responses.

Use exactly the same centres and native consecutive triplets as that adapter.
For each eligible centre, average the three GenD fake-minus-real logits. Build
the same eligible temporal windows: at least three middle centres spanning
0.52–2.6 seconds, two before and two after, no source-centre gap over 0.55 s,
and no triplet spanning over 0.13 s. For each window, let M, L and R be the mean
middle, left and right responses. The candidate statistic is
`min(M - L, M - R) - 0.35 * abs(L - R)`. Sort by descending response, start time
and track ID; retain twelve. A constant classifier bias therefore contributes
no response. This compares learned face appearance over time, not pixel-level
causal attribution and not proof that the mouth is manipulated.

Keep the existing baseline point fixed. Select the candidate's development
point just above the largest non-AI case maximum, floored at zero. Require all
four control conditions; AI labels/responses cannot optimize this point.
Non-AI controls include ordinary local brightening, non-neural crop/resizing,
untouched footage and a 320 ms mux delay. Their zero calibration flags would
be by construction, not a held-out false-alarm estimate.

Evaluate first flagged intervals using the v4 temporal IoU rule of at least
0.2. Unassessed and unflagged positives remain misses. For spatial measurement,
project the actual measured aligned face crop into the representative during
source frame and compare with its exact changed-pixel mask. It is a full-face
measurement region; do not present it as a mouth/pixel attribution. Include zero
IoU for absent flags, masks or out-of-edit representatives. This spatial measure
is secondary: regions of different sizes cannot establish superior precision
merely from one overlap statistic.

The frozen advancement requirements are the same as v4: at least three extra
localized AI cases over the unchanged baseline, at most one lost localization,
at most one extra non-AI flag per control condition, no reduced assessed
coverage, at least 0.01 mean spatial IoU gain over all 16 AI edits, a defined
point and no execution failures. Report both AI edit variants separately and
all 48 terminal outcomes. A failure is rejected without consuming fresh actors
to rescue it. A pass is permission for a separate confirmation design only;
it is not application adoption. No overall-confidence weight is introduced.

All eight actors, the studio environment and the single renderer family are
consumed development. The source/audit catalogue and separate ODbL/DbCL terms
remain applicable. Do not claim outside reviewers have participated. Retain
input failures, code hashes, device/library versions and immutable raw logits.
