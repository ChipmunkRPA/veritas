# AI-only stereo rescue on saved development scores

Compare the previously proposed AI-only rule with mono and a selected-channel-only decision on all 33 forced-imbalance pairs from the completed v282 study. Preserve any existing mono direction. When mono is inconclusive or the gate left it unassessed, accept an AI direction from the selected channel only if that channel was admitted and scored AI. Keep the mono disposition in every other case. A diagnostic score on gate-rejected mono never enters the rule. [Previous paired study](audio-native-stereo-authenticity-v282-result.md)

The readout verifies the exact v282 freeze, development gate records and resource receipt before reading the saved classifier directions. It reports all four outcomes and transitions by family, plus the count of distinct source groups with an added AI direction. The selected-only comparison describes this subset where a physical channel was selected and admitted. No additional inference runs. This is a development replay on reused foregrounds and deliberately imbalanced mixes. The 200-case holdout is a separate prospective comparison.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_ai_only_rescue_v283 freeze
backend/.venv/bin/python -B -m scripts.audio_ai_only_rescue_v283 readout
```
