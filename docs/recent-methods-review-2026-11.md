# Recent research: what changes in this program

This review supplements the existing detection-improvement requirements. It
does not adopt a paper's headline accuracy or authorize restricted code,
weights or training data. Paper ideas, code licenses, checkpoint licenses and
dataset permissions are tracked separately. Reading a method is not a model
replacement; any implementation must pass a fixed matched comparison and fresh
confirmation appropriate to its actual claim.

## Provenance: hard bindings and transformation controls

The October 2026 [soft-binding benchmark](https://arxiv.org/html/2610.04151v1)
separates watermark/fingerprint matching from signed content bindings. Its
protocol fixes false-match rules before robustness tests and audits code,
weight and data permissions separately. This informs the provenance-first
implementation: positive signed claims require asset binding; absent or damaged
signals cannot prove originality. Authored signed-file tests now exercise
metadata loss and content tampering through the actual SDK/adapter, alongside
full pipelines that forbid model acquisition. No benchmark code or weights
were imported. Private keyed-watermark methods remain unavailable unless a
real supported verifier is connected. This is applied software validation,
not measured provider-watermark robustness.

## Audio: preserve the mixture and separate the targets

October's [MixedTruth / TruMix](https://arxiv.org/html/2610.14141v1) keeps original
mixture information while using estimated components as soft evidence. It uses
distinct speech/background streams and structured component predictions rather
than assuming that a separated signal preserves forensic artifacts perfectly.
That matters after our unsuccessful separation comparisons. The next audio
method should retain the original recording, learn separately labeled speech
and background targets, and include genuine/AI combinations of both sources.
Current sound-event categories and noise levels cannot supply synthetic-
background truth. The reviewed paper did not establish a deployable commercial
checkpoint for this application; no code/weights were imported. A new supervised
component method and source-disjoint evaluation are still development work.

## Image: environment invariance needs interventions

August's [EISL](https://arxiv.org/html/2608.17700v1) separates a low-rank feature
projection from its complementary residual and trains with label-preserving
environment interventions. This targets lighting/style shortcuts rather than
assuming a high response proves a forgery. Its guarantees depend on the
interventions and finite data. For our program, paired compression, lighting and
ordinary editing controls must accompany any new feature method. The prepared
48-pair JPEG holdout remains a fixed test of the existing candidate, not a new
training set for this idea. No EISL implementation or checkpoint was imported;
an independent feature-method prototype would be a separately frozen study.

## Video: explain the temporal features and context

October's [explicit forensic-feature method](https://arxiv.org/html/2610.03380v1)
uses tracked face sequences, overlapping temporal windows and four descriptor
families: photometric, textural, geometric and compression. The sequence model
learns their temporal behavior. The useful direction is source-bound,
interpretable measurements with preserved ordering, not copying its reported
scores into our confidence. Our ordinary-edit control already demonstrates why
appearance change alone is insufficient. A next comparison must distinguish
ordinary scene/lighting transitions from face-local changes and retain missing
tracking context. No paper model, code or weights were imported or admitted.

## Passages: genuine web controls and contextual transfer

October's [PAWN++ web evaluation](https://aclanthology.org/2026.wac-1.25/) studies
a compact supervised detector using signals from two frozen language models
and evaluates transfer beyond curated text. This supports testing real web
prose, source/domain shifts, mixed passages and human controls, with each
passage's context and position preserved. It does not validate a whitespace
watermark heuristic or establish reliable authorship for our current article
route. The initial paper review is complete; a full source/checkpoint licensing
audit and implementation comparison remain pending. No weights or code were
imported, and article confidence remains withheld where unsupported.

## Code inspection and execution policy

The [official SynthID Text detector code](https://github.com/google-deepmind/synthid-text/blob/main/src/synthid_text/detector_frequentist.py)
was read alongside its Apache-2.0 license and reference-use restrictions.
Key/configuration-dependent token statistics are different from hidden spaces,
and a null-test p-value must not be displayed as probability of AI authorship.
No production provider key is supplied by a demonstration implementation.
The [provider coverage review](provider-provenance-coverage.md) distinguishes
working local checks, external verifiers and unresolved capabilities.

Restricted research may inform independently written methods. Copying restricted
implementation code or loading restricted checkpoints is a separate action and
does not become permitted because a method was described in a paper. Prefer
commercially licensed components or independently implemented methods with
appropriate data. No new detection model is admitted by this review, and no
claim of improved detection quality follows from reading the papers. Remaining
work includes actual method development, frozen comparisons, fresh controls,
full-app verification and outside human testing.
