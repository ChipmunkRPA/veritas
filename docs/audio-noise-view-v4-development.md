# Noise-robust speech development, v4

Status: development only. The v3 data, all of whose test responses have already
been consumed, can screen an enhancement approach but cannot confirm it.
The running audio detector and its calibration are unchanged.

## Question and source permission

Can a second, noise-reduced view improve voice detection under strong background
sound without increasing false calls on authentic or generated speech? An
improved listening-quality score alone is not evidence of improved detection.
The unmodified recording remains the evidence source. A transformed waveform is
not a recovered original, and enhancement can remove or introduce artifacts.

The first prototype uses the DNS3 checkpoint included in the official
[GTCRN repository](https://github.com/Xiaobin-Rong/gtcrn) at revision
`502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73`. The publisher's root MIT licence
applies to the repository containing source and checkpoint; no separate checkpoint
restriction was present in its recorded tree. Retain Rong Xiaobin's copyright
notice. The 579,819-byte checkpoint SHA-256 is
`a630d992cf792daf4ce2bb5bcf9c4d389f740a8f09c6e0971184697fe6371b79`.
Its Git blob hash was verified against the revision tree. It loads strictly with
`weights_only=True`, with no unsafe checkpoint globals. This records publisher
permission, not an independent audit of all training-source rights or overlap.

DeepFilterNet was also considered. Its README scopes its grant to code, and
upstream checkpoint-licensing questions remained unanswered in issues
[697](https://github.com/Rikorose/DeepFilterNet/issues/697) and
[712](https://github.com/Rikorose/DeepFilterNet/issues/712) when checked. It is
not selected for this commercial-model experiment. No maintainer was contacted.

## Predeclared development panel

Select 20 consumed v3 VCTK identities represented by an authentic source and all
three generated families. Select deterministically by SHA-256 of the identity,
then one source per identity/family by hashed source ID. Add 40 consumed v3
LibriSpeech authentic identities, also ordered by hashed identity. The resulting
120 original files have 60 per class; all six original v3 conditions are kept,
for 720 planned inputs. No source is selected by detector output. Failure stays
in the denominator. Development labels and rights retain the v3 source records.

All variants of a source and speaker stay together. Shared known transcripts and
DFADD prompt indices conservatively connect identity groups for five-fold
cross-validation. Exact DFADD transcript mappings remain unpublished, so even
this grouping cannot establish text independence. Reused backgrounds and
class/source confounding remain explicit limitations.

Enhancement follows the pinned 16 kHz inference: 512-sample STFT, 256-sample hop,
square-root Hann, centered analysis, float32 CPU. Explicit inverse-transform
length retains every original sample position, including lengths not divisible
by the hop. There is no additional clipping or level normalization. The voice
classifier uses its unchanged preprocessing. Its windows and speech gate come
from the unmodified view; unavailable original windows remain unavailable.
No SNR, label, generator, background identity or clean counterpart enters inference.

Before scoring, freeze the exact manifest, source/model pins, implementation
hashes and selection recipe. Score one enhanced view per retained input; do not
search enhancer strengths or substitute sources based on results. Recompute a
small deterministic original-view smoke to verify saved-response parity before
reusing other frozen original responses.

## Analysis and next decision

Compare original-only, enhanced-only and a two-view combination using the same
identity-connected folds, source/class weights and monotone logistic fitting.
Use fixed 0.001 slope regularization, source-count target smoothing and a 50:50
reference prior. Each source shares its weight across conditions. No validation
fold enters its own fit. The serving v3 mapping is a separate operational
reference, not the fairness comparator for a newly refitted method.

Report common-assessed and complete operational denominators, Brier, log loss,
ranking, both false-call directions, abstentions, and class coverage by condition,
source and generator. Diagnostic directional counts reuse the frozen research
operating points; they are not a public service accuracy or threshold claim.
Resample connected source groups for paired uncertainty, conditional on the
consumed background pool. Do not claim 720 independent original observations.

Advance one method only if its overall Brier gain is at least 0.005 and log-loss
gain at least 0.01, its conditional Brier difference interval lies below zero,
and Brier improves in both noise-0 and music-5. At the unchanged research
operating points, each of those strong conditions must gain at least five correct
calls, each condition may gain at most one false call of either kind, and clean
correct calls may decrease by at most two per class. No new execution failure
may explain the apparent gain. Choose the eligible arm with lowest overall
Brier, then log loss; do not select hyperparameters on this panel. Predeclare any exact confirmation
tradeoff before new-source scoring. A changed method requires a new evaluation.
If development does not support an improvement, reject it and preserve the
current service. A promising development result still needs a separately frozen,
fresh multi-generator/voice/background confirmation, own calibration, source-bound
reports and local integration tests before any default or confidence change.

Fresh-source metadata may be inspected for selection and rights without model
inference. Downloaded metadata and prospectively reserved inputs are not a
completed independent test. Published contributor tools do not establish that
an outside reviewer has participated.
