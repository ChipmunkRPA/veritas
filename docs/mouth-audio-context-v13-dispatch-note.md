# Fitting launcher correction before any head training

The scientific implementation and 272-input freeze published in
`42f9b0d5df1e8770912df143fc6f1707e0b43a3e` remain unchanged.
The fitting-source extraction completed all 160 clips/3,560 windows. Before
calling the learning function, command inspection found that the original CLI
uses `fit` for both extraction and learning; its dispatch always selects
extraction. No learning command was attempted and no fitted head was produced.

The separate `train_mouth_audio_context_v13.py` launcher invokes the existing
frozen `fit(trial)` function under the same 900-second/6-GiB/resource-floor guard.
Publish and verify this launcher before its first learning call. It changes no
feature, input, weight, hyperparameter, source role, comparison point or gate.
It does not rerun or overwrite source extraction. Both frozen heads and their
point completion still precede diagnostic audio processing.

Reproduction uses the original CLI for `fit` (audio extraction), `calibration`,
`development` and `score`; use the separate launcher's `guarded` mode for actual
head learning. The original ambiguous dispatcher remains preserved for audit.
