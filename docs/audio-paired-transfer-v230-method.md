# Three-second noisy speech transfer, v230

This development comparison applies the fixed v223 paired-consistency head to the consumed v219 panel. The four-second fitting and calibration objects are used as published. The fixed v226 native extractor supplies a 4,608-value vector from 149 feature frames for each admitted 48,000-sample window. There is no padding, model fitting, or reassessment of the serving speech gate.

The prospective freeze binds the guarded v217 production-waveform preparation, the v219 baseline and resource receipts, the v223 fit and calibration, and the completed v226 native qualification. The five serial CPU batches are matched clean, kitchen 0 dB, kitchen +5 dB, traffic 0 dB, and traffic +5 dB. Each batch retains all 48 planned entries, with one encoder call per originally admitted window. Exact waveform bytes are verified before extraction. Preparation failures and speech-gate rejections remain separate, and neither receives a score. The private journals keep case identities, native feature vectors, digests, and scores. Publish the aggregate only.

Compare the recorded Pella raw output with the fixed head's sigmoid output on common assessed cases. For each condition, count concordant, tied, and discordant synthetic–genuine pairs for all synthetic cases and for each source generator, including a separate same-triplet comparison. Report clean-to-noise drift for the same foreground and each method on both score and bounded logit scales. Comparisons share source voices and noise tapes, so the resulting counts are descriptive and do not establish independent population error rates.

Show the original v219 directions separately from experimental directions at each method's frozen v223 English score points. The old route uses the recorded English-mapped score for this diagnostic. The paired head uses its new feature-derived score. These points measure transfer behavior and do not give calibrated confidence for three-second Turkish material. The study leaves product behavior unchanged.

From the project root, use the existing backend environment.

```sh
export PYTHONPATH=backend
backend/.venv/bin/python -B -m scripts.audio_paired_transfer_v230 preflight
backend/.venv/bin/python -B -m scripts.audio_paired_transfer_v230 freeze
for batch in 0 1 2 3 4; do
  backend/.venv/bin/python -B -m scripts.audio_paired_transfer_v230 run "$batch" || break
done
backend/.venv/bin/python -B -m scripts.audio_paired_transfer_v230 readout
```
