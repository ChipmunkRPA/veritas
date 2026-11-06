# Original-waveform quality calibration, v5

Predeclared development only. The running voice detector and its fixed v3
calibration remain unchanged. No new pretrained model is introduced. Existing
voice and speech-gate responses retain their pinned commercial-permission
records. This is a new calibration method, not a claim about noise removal.

## Question and fixed design

Can observable original-waveform properties help interpret the same voice
detector response under background sound, without reducing useful conclusions?
The v4 enhancement approach failed its development gate. This experiment does
not reuse its transformed-waveform scores or choose another denoising strength.

Use exactly the already-consumed v4 development panel: 120 original files,
60 authentic and 60 generated, with six conditions for 720 planned inputs.
Keep its five identity/prompt-connected folds, source/class weights and complete
failure denominators. There are 60 connected groups, not 720 independent voices.
The original v3 source records and raw responses are frozen by their hashes.

Compute two descriptors from the unchanged production-decoded 16 kHz float32
waveform. Divide a working copy by peak only to make the descriptors insensitive
to gain; never alter the classifier input. Use noncentered 512-sample frames with
256-sample hops. Spectral flatness is the median geometric/arithmetic power ratio
across 80–7600 Hz, using a periodic Hann window and a per-frame relative 1e-12
power floor. Map it to [0,1] as clip((log10(flatness)+5)/5). The second coordinate
is clip((log10(p10/p90)+4)/4), where percentiles concern unwindowed frame energy.
Zero-energy or invalid recordings cannot acquire a probability from these
descriptors. Neither coordinate is measured SNR, a proof of editing, or an
independent authenticity check.

Compare one candidate against a fairly refitted original-only logistic mapping.
The candidate uses raw logit, both coordinates, their two interactions with raw
logit, and an intercept: six parameters. Constrain the coefficient of raw logit
to be nonnegative at every corner of the [0,1] quality box, ensuring monotonicity
throughout. Use fixed slope ridge 0.001, source-count target smoothing and a
balanced reference prior. No hyperparameter sweep or condition/generator/source
metadata enters prediction. Every held-out source uses parameters fitted on
the other four folds. Keep the deployed guarded calibration as a separate
operational reference; do not label a refit as the current running service.

Fit a quality-only nuisance diagnostic on the same folds, excluded from candidate
selection. Strong discrimination from quality alone can reflect source/channel
confounding rather than forensics. Report the shared-voice VCTK/DFADD panel
separately, plus generator, source dataset and individual background partitions.
Neither matched identities nor this diagnostic establishes pretrained or prompt
independence. Unpublished DFADD text mappings remain a limitation.

## Advancement decision frozen before extraction

Reuse the v4 fixed research operating points solely for directional diagnostics;
they are not public threshold or accuracy promises. Require all of:

- Overall common-assessed Brier improvement at least 0.005, log-loss improvement
  at least 0.01, and conditional paired Brier interval entirely below zero.
- Better Brier score in both noise-0 and music-5, and in the joint-voice panel.
- At most one additional false AI and one additional false original per condition.
- At most two lost clean correct calls per class.
- At least five additional correct directional calls in each noise-0 and music-5.
- Identical assessed inputs and no new feature-execution failures.

Use the inherited 2,000 connected-group bootstrap draws with fixed seed 461109,
stratified by joint VCTK versus Libri identity groups. Intervals condition on
fixed fold fits and reused backgrounds, and exclude refitting uncertainty.
Report abstentions as well as both error directions. Do not turn missing inputs
into original judgments or omit them to improve the apparent operational rate.

Freeze the implementation, protocol and manifest before extracting features.
If any gate fails, reject the method and retain the incumbent. Passing is only
permission to design a separate, frozen fresh confirmation with its own fitting
and calibration sources. It is never deployment permission. Fresh multi-generator
audio reserved for a viable method must not be scored to rescue failed development.
