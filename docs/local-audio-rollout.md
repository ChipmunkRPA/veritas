# Local audio replacement

This is an explicitly requested local experimental release, separate from the
stricter research admission decision. The frozen v3 study and its 26 unmet gates
remain unchanged. This release does not claim that noisy speech, unfamiliar
languages, or longer recordings have passed general deployment validation.

## Selection and rollback

`VERITAS_AUDIO_DETECTOR_MODE=candidate` selects the candidate for fresh local
analyses. `legacy` restores the previous audio detector and its existing fusion
calibration. Restart the backend after changing the setting. The repository
setting defaults to `legacy`; the local `.env` is switched only after integration
verification. Neither the old checkpoint nor its calibration is removed.

Use the matching mode when reproducing an old analysis. Existing saved results
remain immutable and are interpreted using their saved calibration identity.
The candidate calibration is not automatically promoted by an evaluation script.

## Confidence calculation

The serving bundle is `backend/app/detectors/audio/calibrations/local-speech-v3.json`.
Its SHA-256 is `912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf`.
The source calibration SHA-256 is
`914f34e21f9aa670d800aec1a055d9b6f9e7a5734de3fed6cf886c1dd246c15b`.
Both identify fixed artifacts, not newly fitted values.

The unrounded candidate response uses its own monotone logistic calibration,
with slope 0.3955898971191526, intercept 0.13561941517712647, and a balanced
reference prior. It never enters the old detector's probability mapping.
The 80% tendency threshold also requires the frozen voice-group uncertainty
check. An original tendency additionally requires the fixed bandpass stability
probe; a missing or unstable probe withholds that conclusion.

The validated input format for this local reference calculation is one 2.5–4
second speech window, as used in the English evaluation. Language is not
automatically identified. Longer recordings still receive window-level speech
and background-sound evidence, but no calibrated whole-recording confidence.
Silence receives no authenticity probability. Background tags and waveform
changes provide context; they do not independently prove synthetic speech.
The percentage is a reference-calibrated screening estimate, not a measured
accuracy rate for the customer's recording. No per-file confidence interval is
invented.

For video, the new speech result is reported separately. The existing visual
fusion uses its fitted missing-input feature for the retired voice slot; the
candidate does not substitute a new score into that slot or enter the combined
video percentage. All other compatibility checks remain enforced.

## Repeatable local verification

Run `backend/.venv/bin/python -m pytest backend/tests/test_local_audio_rollout.py`.
With the installed models and frozen local research fixtures, run:

```sh
backend/.venv/bin/python backend/scripts/validate_local_audio_rollout.py --mode candidate --output output/audio-rollout/candidate-check
backend/.venv/bin/python backend/scripts/validate_local_audio_rollout.py --mode legacy --output output/audio-rollout/rollback-check
```

Choose a new output directory for each run. The script uses isolated databases
and a simulated Stripe HTTP transport. It verifies that a signed paid receipt
is required before a job exists, then runs the actual worker and model weights.
It checks clean and strongly noisy authentic/AI speech, long recordings,
silence, and a video with an AI speech track. It validates PDF signatures,
JSON/evidence correspondence, exact waveform image pixels, media ranges,
source coordinates, public model-name filtering, and the offline evidence
package. These are engineering checks on consumed fixtures, not a new accuracy
benchmark or a real Stripe charge. Customer billing remains enabled.

## Completed local verification (31 October 2026, Pacific time)

Seven candidate engineering cases and two legacy rollback cases completed real
model inference. Six standalone audio cases and the video speech case passed
PDF/JSON/evidence/signature/package verification. The independent voice estimate
matched the frozen calibration calculation; all 1,800 saved test predictions
also passed inference-only numerical/decision parity without refitting.

The selected clean original clip returned a 93.3% original tendency; the selected
clean AI clip returned a 95.5% AI tendency. Both 0 dB noisy clips were
inconclusive, as were the long recording; silence was not assessable. These
examples are plumbing checks, not new accuracy statistics. The video fixture
used a plain slate with AI speech; face checks encountered the memory budget,
which was disclosed, and its whole-video result stayed inconclusive.

The local backend now runs in candidate mode. The actual customer API still
requires prepaid checkout, at USD 9.99 in Stripe test mode. Verification
receipts are in `output/audio-rollout/verification.json`; current audio exports
are in `output/audio-rollout/verified-exports/`, the video export in
`output/audio-rollout/verified-video-final/`, and rollback receipts in
`output/audio-rollout/rollback-final/`. Earlier failed verification attempts
remain separate and were not relabelled as successful.
