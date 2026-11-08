# Recent research: what changes in this program

This review supplements the existing detection-improvement requirements. It
does not adopt a paper's headline accuracy or authorize restricted code,
weights or training data. Paper ideas, code licenses, checkpoint licenses and
dataset permissions are tracked separately. Reading a method is not a model
replacement; any implementation must pass a fixed matched comparison and fresh
confirmation appropriate to its actual claim.

The [8 November passage/mixed-audio source review](passage-and-mixture-method-review-2026-11-08.md)
adds direct generated-span prediction and independently labeled speech/event
mixtures. It pins inspected inference, matching, coordinate and loss files.
The audio repository's permissive code and noncommercial checkpoints remain
separate; those weights are excluded. The passage checkpoint's large backbone,
data rights, offset handling and independent calibration still need
qualification. Neither reading changes serving or supplies an accuracy result.

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
ordinary editing controls must accompany any new feature method. The 48-pair
JPEG confirmation is now complete: the candidate failed three fixed gates and
was rejected. Those sources are consumed development data, not another fresh
holdout. No EISL implementation or checkpoint was imported;
an independent feature-method prototype would be a separately frozen study.
That independent [noise/texture intervention study](image-residual-v13-result.md)
is now complete on consumed development sources: both simple heads lose
localization quality and are rejected. It neither reproduces nor tests EISL's
learned CLIP/LoRA implementation, and imports none of its code or weights.
The subsequent [supervised regional head](image-patch-head-v14-result.md) is
also complete: frozen licensed visual features plus a small learned head reduce
false highlights but lose useful localization, so the candidate is rejected.
Its strict mean-precision audit failed and remains failed. A separately frozen
full native diagnostic replay reproduces all original features/maps and outcomes;
the accurate-mean sensitivity probe changes no selected masks. This narrows the
next work to stronger regional methodology and precision-qualified context
features, rather than adopting the candidate from fitting loss or a subgroup.
The [magnified-crop comparison](image-iml-zoom-v16-result.md) separately tests
the existing permitted transformer localizer at larger crop scale. All384 input
variants complete, but useful localization is0/48 per format versus9/48 native
and6/48 JPEG current. Rejected. Its original exact independent-mask audit fails
at rounding boundaries; a complete post-result sensitivity analysis retains
the discrepancies and confirms rejection, without clearing the failed audit.

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

The [subsequent factorial image experiment](image-region-point-v15-result.md)
separately tests accurate context means and calibration of actual displayed
components. It passes the complete new native arithmetic audit but fails pooled
localization improvement, so no arm is adopted. The
[broader signal search](provider-signal-broad-search-2026-11-08.md) also adds
CapCut-family verification, hosted Cosmos marking and other scoped source
records without presenting them as admitted detectors.

The [scene-context comparison](video-background-context-v25-result.md) now tests
a concrete source-bound surrounding-pixel descriptor alongside face transitions.
All 280 planned files and 600 outcomes completed. It removes two ordinary scene-
substitution alerts but adds no swap beyond appearance recalibration and fails
the unchanged replacement gates. Coarse background colour is insufficient; no
new confidence or serving cue was admitted.

The [current permission review](lip-sync-source-review-2026-11-08.md) excludes
unresolved pretrained lip-sync assets without substituting missing weights.
The resulting [LTX public-pattern audio study](ltx-marking-audio-v1-result.md)
completes all 80 cases and a separate full audit; wrong-payload presence and
noisy/partial misses remain retained. The expanded
[provider review](provider-signal-followup-2026-11-08.md) also reads textGrain
generation/detection mathematics while preserving its secret-key dependency.
These are distinct from adopting a new lip-sync or passage detector.

The independently implemented [sound/mouth correspondence study](mouth-audio-context-v14-result.md)
adds actual clock-bound audio measurements to the existing temporal visual
learner, with an unchanged-capacity visual-only control. It finds five additional
known edits but doubles ordinary flags, including original and merely delayed
recordings. Its full source/audio/feature audit passes; the method is rejected.
Correlation measures correspondence, not AI provenance. No new paper weights,
customer confidence or model default was introduced.
