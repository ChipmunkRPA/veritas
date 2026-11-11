# Rollback voice classifier on the shared river comparison

The [v214 crossover](audio-demand-crossover-score-v214-result.md) found that
the incumbent lost AI flags under two excerpts of the same river recording
when both classes heard the same excerpt. The frozen
[WavLM complement](audio-wavlm-v13-protocol.md) and
[combined adapter](audio-combined-serving-v97-result.md) use exactly
64,000 samples. None of the saved prepared v208 or v210 inputs reaches that
extent. Keep the original shorter windows and test the cached rollback
classifier as a separate development diagnostic.

Use the Apache-2.0 Gustking snapshot at revision
`f7050b586236dc910d1157f430def2d0647b02b4`. Bind the three snapshot
assets to the app manifest and verify the 16 kHz preprocessor, normalized
input, attention mask and index-one `fake` label. Load the exact local
snapshot with the standard Transformers classes and use the application's
existing single-window classifier preprocessing. Record the CPU float32
runtime. [Pinned publisher snapshot](https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification/tree/f7050b586236dc910d1157f430def2d0647b02b4)

Keep all 144 planned v210 slots in the clean, early river and late river
conditions. Bind the verified v210 production WAVs and the v214 guarded
private scores and public aggregate. Score only the original route's
admitted complete windows, one rollback forward each. Keep recorded
unavailable slots and gate rejections. There is no padding, conditioning,
additional speech gate, or fresh-panel access.

On identical assessed inputs, report cross-class rank concordance,
ties, and concordance within exact paired excerpts. For each model,
report paired raw score movement from its own matched-clean view.
Preserve the incumbent's recorded decisions separately. Count scores on
the fake side of the rollback adapter's existing raw 0.5 comparison
point as a diagnostic, also showing the incumbent raw scores at that same
point for context. These scores have no transferred calibration or new
customer decision. The comparison uses consumed Turkish foregrounds,
one generator collection and one river recording. A useful rank gain
would motivate separate calibration and fresh confirmation.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_rollback_crossover_v220 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_rollback_crossover_v220 freeze
for batch in 0 1 2; do
  PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_rollback_crossover_v220 run "$batch"
done
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_rollback_crossover_v220 readout
```

Workers use the unchanged v108 offline 6 GiB neural admission guard.
The readout verifies guarded case and profile commitments and returns
aggregate results. Individual scores, paired identities and WAVs stay
private. Serving settings and confidence are unchanged.
