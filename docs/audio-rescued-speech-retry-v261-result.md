# Existing voice scoring on six recovered genuine clips

The existing voice classifier scored all six genuine noisy clips recovered by the speech-admission fallback. The clips came from five source speakers. Two leaned original, four were inconclusive, and zero leaned AI. The clips that leaned original were one kitchen and one traffic variant, both at 0 dB SNR. The worker completed eight classifier forwards, including two stability probes. [Exact result](evaluation-panels/audio-rescued-speech-retry-complete-v261.json) · [Admission results](audio-noise-speech-gate-v256-result.md)

The initial scoring worker stopped before model loading because it initialized CPU interop threads twice. The retry used the existing loader for that initialization and verified the same six frozen inputs and earlier failed receipt. The retry scored the original whole windows with the existing calibration. [Retry method](audio-rescued-speech-retry-v261-method.md)

These results support a fresh test of the speech-admission fallback. This replay uses the same development cohort that revealed the coverage loss. Independent background and speaker tests are still needed to estimate its errors. The six clips were tested with the existing classifier only.
