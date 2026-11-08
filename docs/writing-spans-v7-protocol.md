# Controlled writing span learning, v7: prospective local protocol

This is a development comparison of passage localization, not a release of a
new detector or an accuracy claim for customers. Publish this protocol, code,
and aggregate commitments before the first project-source model forward or
head fit. All computation and replay are local. GitHub Actions remains disabled;
publication uses direct pushes following local checks.

## Question and licensed ingredients

Can a small supervised interval learner locate known inserted AI source slices
better than both a matched token learner and the current three-sentence-context
passage check, at a similar genuine-source flag rate? A plain classifier's
whole-passage response is not itself a pixel- or word-origin map.

Use the cached Apache-2.0 checkpoint
`GeorgeDrayson/modernbert-ai-detection-raid-mage` at
`e047d9c56f51a7e5c5ef5b0dd556f519bb9dd624`, based on the Apache-2.0
ModernBERT-base encoder. The pinned cached model card, base card, assets and
actual standard state loader are hash bound. No weights or code are downloaded
in this study. Its RAID/MAGE development history prevents an unseen-training
claim. The cached fine-tuned card could not be retrieved again by the web tool;
that does not alter or supplement its cached license statement.

Use the existing Apache-2.0 four-class small text checkpoint, preserving its
upstream MIT notice, as the current primary passage comparator. Its raw
`1 - human` response includes rewriting classes; it is not a calibrated AI
origin probability. This is not a comparison to the complete SaaS fusion.

TXD-22 v1, DOI 10.17632/prcjcggtjf.1, is offered under CC BY 4.0 by its
[publisher](https://data.mendeley.com/datasets/prcjcggtjf/1). Attribute the
publisher and source in results; do not redistribute the source text or these
private compositions. The exact original 66,000-row CSV, its 22 classes, all
connected prompt/answer components, and the previously selected 128 groups
are checked again. AI labels and the publisher's human labels are not
independent authorship verification. Authors are unidentified and overlap with
external training data is unknown. The old development outcomes are consumed.

The finer-grained AITDNA corpus is not used for new training because the
institutional and bundled terms need clarification. No noncommercial
checkpoint, large new backbone, outside service, or customer upload is used.
GigaCheck's published interval-query approach informs the task formulation;
this small implementation is our own, without its code or weights. See the
dated [method review](passage-and-mixture-method-review-2026-11-08.md).

## Fixed source construction and roles

Map the old v5 roles `specialist_cal`, `fusion_fit`, `fusion_cal`, and
`confirmation` to **fit**, **cal**, **point**, and **diagnostic**, respectively.
Each role has 32 prompt/duplicate components. Components, original answers,
and constructed exact-text duplicates cannot cross roles. No author-level or
fresh-source independence is asserted.

Each fitting, calibration, or diagnostic group has nine deterministic
160-whitespace-word compositions: pure human, pure AI, a 12-word AI insert,
a 60-word AI insert, a 12-word human gap in mostly AI, three separate AI
regions, and three matched all-human rearrangements. Copy the original ASCII
word slices exactly, preserving internal punctuation and spacing; insert only
two newlines at slice joins. Hashed source-group identifiers choose the two
insertion locations. The `recipes` implementation specifies every boundary.
The point role contains only the four genuine compositions: **no AI examples
are inferred to choose a point**.

Totals: fit288, cal288, point128, diagnostic288 = **992 documents**. There are
768 source-AI intervals across the three roles containing AI, not 768
independent authors. Known-source non-whitespace Unicode character positions
are the target; whitespace is excluded from all counts. These are deliberately
constructed slices, not natural article/email or intellectual-authorship truth.
Source text, raw/extracted mappings, row pairings and learned heads stay private.

## Native runtime and two own learners

The frozen 149,606,402-parameter classifier supplies its complete last native
encoder layer as 768-wide CPU float32 vectors. Two threads, eager attention,
deterministic algorithms, 512-token stride-zero overflow windows; no truncation
or omitted non-whitespace character is allowed. Capture actual input tensors,
all offsets, complete feature hashes, and audit the feature tap against the
full classifier on the first document per role. Arrays remain **in RAM only**.
The current comparator similarly binds actual tensor inputs, complete
three-sentence contexts and focus locations; first-role and all overflow
contexts receive a separate unbatched response check (absolute error <=1e-5).

An authored runtime probe first failed because a combining accent was not
covered by the native tokenizer; preserve that failure and its resource log.
The revised authored probe explicitly requires that unsupported input be
refused **before any new forward**, alongside supported ASCII and long-overflow
probes. This is an execution check, not detection quality. The token-only
preflight found no omitted non-whitespace characters in all992 planned texts.
It does not validate other Unicode input populations.

Fit only on the 288 fitting documents. The token control is a 768-to-one
linear head with soft within-token AI source-character fractions and
non-whitespace-weighted per-document BCE. The interval head projects to64,
adds fixed sine/cosine normalized character-position encoding, and uses eight
learned queries with two Transformer decoder layers, four attention heads,
128-wide feedforward layers and zero dropout. Predict foreground/background
and normalized center/width. Match ground truth one-to-one by Hungarian
assignment with class,5xL1 and -2x generalized IoU cost; train weighted CE
(background.1),5xL1 and2x(1-GIoU) losses. Pure-human documents have empty targets.
Interval readout is floor(start x text length), ceil(end x text length), using
the actual stored float32 endpoints. No quantized score raster is substituted.

Seed20261108, batch8, AdamW decay.001, gradient clipping1. Token40 epochs
at.003; interval60 epochs at.0005. Use the final epoch only. Never select
epochs, hyperparameters or architectures using diagnostic outcomes. Save
every fitted parameter and epoch loss privately. Pretrained parameters must
remain frozen, without gradients, and exact checkpoint files unchanged.

## Separate calibration, genuine points, and sealed diagnostics

For each arm separately, map raw proposed-interval responses to the retained
publisher-AI non-whitespace fraction *within that proposed interval*, using
only the288 cal documents. Equal document/group mass; proposals divide their
document's mass equally. Use nonnegative logistic calibration, raw clipping
1e-6, weighted logit standardization, slope penalty.001, L-BFGS-B(maxiter500,
maxls50,gtol1e-9,ftol2.22e-16). Require reported convergence and projected
gradient <=1e-6. No class-prior transfer or customer-confidence claim follows.

On the128 point documents, first take each document's highest calibrated
proposal score, then the maximum across its four genuine compositions.
For each arm independently, choose max(.001,nextafter(the second largest
of32 group maxima,+infinity)). This allows at most one flagged group in this
particular point set, including ties. It is not a validated population false
alarm rate or a public screening threshold. Seal fitted heads, calibration,
point journals and points with hashes **before** the288 diagnostics.

Score all diagnostics once. A highlight is the union of all proposed source
intervals at or above that arm's fixed point, excluding whitespace. Charge
all highlighted human characters and missed AI characters. Report TP/FP/FN,
character IoU/F1, and recovery of each known source-AI interval when at least
half its non-whitespace characters are highlighted. A useful document
location requires IoU>=.3. Pure AI,12-word AI, long AI, human-gap, multi-region,
unchanged human and ordinary human rearrangements remain separate.

The primary interval learner must beat **both** current and matched token:
more useful mixed-document locations, higher mixed mean IoU, no loss of a
previously useful mixed location, no fewer useful pure-AI or short-AI documents,
and no fewer recovered known intervals. Genuine source-group flags overall,
and flags in each of the four genuine conditions, may increase by at most one.
Every planned case must complete and the strict local replay must pass.
Report a paired source-group bootstrap95% interval for mean mixed IoU
differences (2,000 resamples,seed20261208), without treating9 compositions as
independent people. Failed gates remain failed; no post-result relaxation.

**Even a successful development comparison cannot enter serving confidence.**
Fresh, independently attributable source data and broader qualifications are
still required. Keep all current serving defaults, old-model rollback,
payment behavior and public report confidence unchanged.

## Local replay and resource/publication boundaries

After diagnostics, separately reconstruct all992 documents from the original
CSV and source graph. Re-infer every complete feature hash and all current
responses; require exact positions/proposals/calibrated scores. Retrain both
small heads using only fitting features and require exact saved states and
epoch losses; refit calibrators and genuine points exactly. A separate
set-based implementation recounts all864 diagnostic arm masks. Recompute
summary and frozen gate decisions. This is local execution review using the
same native libraries, not an outside researcher or independent validation
population. Count every replay forward and retain unsuccessful attempts.

Run heavy phases sequentially under a900-second,6-GiB sampled process-tree
RSS guard with at least1536MiB available RAM and disk. Preserve sampled RSS
as a lower bound, every sampling error, failure and incomplete denominator.
No overwriting/restarting a failed phase, no silent fallback. A failure needs
a separately documented prospective successor, rather than changing this lock.

Publish only code, protocols, hashes, aggregate metrics and limitations by
direct push after local verification. Do not publish source texts, private
pairings, vectors or learned heads; do not start GitHub workflows. No new
external collaboration, cloud resource or billing action is authorized here.
