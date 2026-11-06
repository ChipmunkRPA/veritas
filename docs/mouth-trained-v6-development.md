# Direct mouth-artifact head: development protocol v6

This is a new learned representation, not a relaxed point for rejected v4/v5
statistics. No serving detector or overall confidence changes during research.
All 28 source actors are already consumed from prior timing work. Eight used
in v4/v5 remain development only; they are not fresh confirmation.

## Source-separated fitting

Use all 20 remaining original CREMA-D recordings, excluding the eight v4 actor
groups and source-byte identities. Freeze them by SHA-256 order with seed
`chipmunk.mouth-trained/6`; the first 16 actors fit the head, the last four
choose its review point. Keep actor/source variants together. Do not select
recordings using any detector response. Each has 50–100 complete common
25 Hz picture/16 kHz audio frames. Training uses native still frames and needs
less context than the 63-frame video comparison; it does not pad or loop them.

For each source, use a centred 20-frame edit interval and eight fixed positions
0,3,5,8,11,14,16,19 within it. Construct five variants of each selected frame:
original, local 12% brightening, non-neural resampling, same-frame AI mouth
redraw and eight-frame-earlier AI mouth retargeting. Reuse the pinned permitted
human portrait renderer, MediaPipe geometry, mouth support and edge opacity
from v4. The two neural variants are AI irrespective of timing agreement.
Record exact changed pixels and unchanged outside support. Lossless PNGs must
decode to the planned native arrays. This gives 800 planned labelled frames,
not 800 independent subjects: 640 fitting and 160 review-point frames.
Any unavailable case remains recorded; do not train a candidate on an incomplete
plan or silently choose another actor/frame.

The source's ODbL/DbCL terms, generator's declared MIT and permitted replacement
geometry from v4 remain applicable. The feature checkpoint is the installed
Apache-2.0 DINOv2-small revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`,
weight SHA `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
Pretraining overlap and a complete training-rights audit are not established.
No restricted upstream face detector is used; production YuNet supplies the crop.

## Fixed representation and fit

Detect and align the largest eligible face using unchanged production YuNet and
GenD alignment. Select the native aligned rectangle x=[0.28,0.72], y=[0.60,0.92]
using floor/ceil integer bounds, then resize that mouth/lower-face chip to
112 square with bicubic interpolation. Use ImageNet normalization. A frozen
DINOv2-small produces 384-dimensional layer-normalized CLS and mean patch
vectors, each L2-normalized; concatenate to 768 features. Batch size is twelve,
float32 MPS. Record source, native/face/chip/tensor hashes and exact transforms.
The chip is a measurement region, not causal attribution or a changed-pixel mask.

Fit a single L2 logistic linear head with C=0.1, lbfgs, max_iter=2000, tol=1e-8,
seed zero. Feature centring/scaling uses fitting actors only, weighted mean and
variance with scale floor 0.05. Give each actor equal total weight and each class
half of it, distributing weight equally across that actor's conditions and
selected frames within class. Rescale weights to sum to the fitting row count.
Do not tune architecture, regularisation, region or polarity using development
or point-selection actors. Non-convergence is a failed candidate.

Responses are raw signed margins, never calibrated AI confidence. Select the
review point just above the maximum non-AI margin among the four point-selection
actors, with a zero floor. Do not use their AI responses to optimize it. Preserve
fit/point actor IDs, feature receipts, head coefficients and all weight/code
identities. This is not a probability calibrator or a public screening cutoff.

## Development comparison and gates

Before feature extraction or fitting, freeze its implementation and this design.
Generation may be separately frozen and run first because it inspects no new
detector response. For evaluation use exactly the existing 48 v4 clips, including
all 16 known AI mouth edits and 32 ordinary controls. Verify unchanged production
baseline windows and source centres against their immutable receipts. Use the
same tracked centres and native triplets; candidate frame margins are averaged
over each triplet. Eligible windows retain the v4 source-context rules: two
before/two after, middle span 0.52–2.6 s, source-centre gaps at most 0.55 s and
triplet span at most 0.13 s. Rank windows by mean middle margin, then start and
track ID, retaining twelve. The point remains fixed from the separate four
actors. No development control maxima may replace it.

The first flagged window needs temporal IoU at least 0.2 with the known edit.
Project the actual measured chip at the chronological middle during centre and
compare with the exact changed-pixel mask. Missing flags or out-of-edit
representatives receive zero overlap. Keep all planned cases in denominators.
Advance only with at least three additional localized AI edits, at most one lost
baseline localization, at most one extra non-AI review flag per control
condition, no reduced assessed coverage, at least 0.01 mean spatial IoU gain
over all 16 edits and no execution failures. Record both AI conditions separately.
Exact source/crop/tensor replay and a frame-ordering regression are required.

Failure is rejected without consuming fresh actors to rescue it. A pass allows
only a separately frozen confirmation design with unseen actors AND sentence
families, realistic ordinary edits and a second permitted editing family.
Independent outside review, UI/PDF/JSON integration, paid-flow testing and any
overall-confidence calibration remain separate requirements. None has been
satisfied merely by fitting this experimental head.
