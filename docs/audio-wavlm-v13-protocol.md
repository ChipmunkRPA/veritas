# Noise-aware speech backbone: prospective comparison

6 November 2026. Test a different representation after fixed-layer v12 failed
the original useful-detection requirements. No project-source forward or fit
precedes this protocol's implementation/source/runtime freeze. GCP remains last.

## Source and hypothesis

Use Microsoft's `wavlm-base-plus`, revision
`4c66d4806a428f2e922ccfa1a962776e232d487b`, 94,381,936 frozen parameters.
Publisher LFS SHA-256 is
`3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`.
The [model card](https://huggingface.co/microsoft/wavlm-base-plus/blob/4c66d4806a428f2e922ccfa1a962776e232d487b/README.md)
links its license to [UniSpeech CC-BY-SA-3.0](https://github.com/microsoft/UniSpeech/blob/6112826ac13a4327f4c9a7afa2a505e35b763514/LICENSE).
Retain attribution/share-alike obligations for any covered distribution/adaptation;
do not substitute the separate unilm code's MIT license for that weight reference.
No full checkpoint or third-party training corpus is redistributed here.

The [paper](https://arxiv.org/abs/2110.13900) describes masked prediction,
denoising and utterance mixing. We hypothesize those representations may help
distinguish synthetic speech under interference. This is an inference about a
candidate, not measured deepfake performance. It is not a pretrained AI detector.
English pretraining limits and unknown overlap with Libri-Light/GigaSpeech/
VoxPopuli remain; disjoint project roles do not prove pretraining disjointness.

## Fixed extraction and learner

Use exactly the original 4,248 consumed four-second mono16kHz float32 inputs,
with original roles 1,926 fitting / 873 calibration / 1,449 development and all
62 original unassessed statuses retained. Known original/cloned speaker groups
remain together. The reserved1,080 confirmation variants remain excluded.
Rehash every waveform before freeze and validate decoded PCM before extraction.
No padding, resampling, source separation, denoising or amplitude normalization.
Preserve the upstream `do_normalize:false` preprocessing declaration. CPU float32,
eager attention, eval, strict weight-only load and all parameters frozen.

Return all13 hidden states. For indices4/8/12, concatenate population mean and
standard deviation over the199 valid frames (768 wide), in layer/statistic order.
This supplies4,608 fixed features, source/feature digests and exact convolution
clock (stride320 samples, receptive field400). There are no source labels,
generator identities, recording conditions or reference originals as predictors.

Two fixed arms: `standalone` fits these features plus intercept; `complement`
adds the current detector's clipped raw logit. Both learn fitting-only weighted
center/std (floor1e-6), scaled by sqrt(16/4608), followed by the same weighted
smoothed logistic objective and fixed L2 as v12: representation0.05, raw0.001,
intercept0. Complement's explicit raw coefficient is nonnegative; correlations
between features do not establish physical monotonicity. Standalone does not
use the raw score as a predictor, but retains the current assessed scope and
raw stability safety check. It is therefore not yet an independent full service.

Use the unchanged L-BFGS-B tolerances/iteration cap and balanced
truth/family/group/source/assessed-variant weights. Both receive their own
independent worst-group/truth calibration. Preserve missing calibration values,
original internal research decision points and raw/candidate band-filter stability
checks. Research operating points are not public screening claims. Save fitted
objects before calibration; never refit normalization/coefficients on calibration
or development. Non-convergence stops the arm; retain the failed attempt.

## Frozen comparison and decision

Compare all1,449 development outcomes against exactly the recorded current
service; unassessed/failures cannot become original conclusions. Reconstruct
the original sixteen adoption gates without changing allowances. Report complete
condition/generator strata, correct AI/original calls, false calls, missed AI
including abstentions, inconclusive rates, proper probability scores and paired
connected-source uncertainty. All source variants remain one statistical group.

Only arms passing every gate can advance. Prefer standalone if both qualify,
unless complement gains at least five correct conclusions with neither false-call
direction higher and Brier no worse. Passing development alone cannot enable
serving. Failure rejects without using the reserved confirmation panel. Do not
adjust features, ridge, decision points or guard rules to rescue a failed result.

Before any fresh confirmation, independently reconstruct fitting/calibration/
probability/operational arithmetic, exactly refit both heads, and replay all99
predeclared development cells plus every newly required filtered probe natively.
Keep original model/raw probe parity when extending the filtered-probe cache.
Fresh confirmation requires a separately recorded prospective freeze and complete
outcomes. Appropriate source-scope validation, UI/PDF/JSON evidence, name filtering,
prepaid-flow tests and rollback precede any local adoption or confidence admission.

Public method metadata and code do not establish outside participation. Source
waveforms, private paths, transcripts and customer uploads stay private. Publishing
new derived feature packets is a separate pending disclosure decision. No change
to billing, service defaults, whole-file confidence or cloud resources is authorized
by this experiment's completion alone. Image/video/writing and actual outside
testing remain unfinished requirements of the wider improvement program.
