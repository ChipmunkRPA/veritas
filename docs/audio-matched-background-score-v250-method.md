# Frozen local second-opinion comparison on the matched audio panel

Replay the completed v248 materialization and compare its published aggregate, frozen inputs and private case journal. The reconstruction runs in a separate data-only process so v219's scoring process starts without application settings or model imports. Reuse the serving WAV reader to check each of the 360 planned rows' available signed-16-bit inputs and bind its native float32 sample digest. Keep unavailable preparations in their original positions. Carry forward source group and recording reuse counts, with one Okur stock voice and provisional machine-screened background evidence. [Matched preparation](audio-matched-background-panel-v248-method.md)

Use the unchanged v219 serving speech route, including its original activity gate, raw classifier and stability decision. Run five baseline batches in fixed condition order, each with 72 planned rows. Verify all five baseline receipts before any candidate call. Then run five candidate batches in the same order. A candidate call requires the exact serving-gate-admitted full 48,000-sample window. Use the frozen v226 native encoder and v230 transform with the unchanged v223 fitted head and English diagnostic points. Check the candidate feature packet's input digest against the baseline's frozen float32 samples. Run the existing local v108 resource guard in each separate phase and preserve unfinished or unavailable work. [Serving comparison](audio-fresh-shared-noise-score-v219-method.md) · [Frozen candidate method](audio-paired-transfer-v230-method.md)

The second opinion changes a baseline inconclusive direction to AI only when the candidate's frozen English diagnostic direction is AI. All baseline AI, original and unavailable directions retain their recorded values. Report recovered synthetic variants and added genuine false-AI variants separately by condition, arm and original source cohort, along with planned coverage and shared-gate rejections. Compare raw-score rank concordance for each generator against genuine speech on all assessed pairs and on matched triplets. Preserve the one-stock-voice and repeated-recording denominators. Treat the English points as diagnostic, with no fitting, Turkish confidence calibration or serving changes. [Earlier second-opinion replay](audio-paired-transfer-v230-result.md)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.audio_matched_background_score_v250 preflight
backend/.venv/bin/python -B -m scripts.audio_matched_background_score_v250 freeze
for phase in baseline candidate; do
  for batch in 0 1 2 3 4; do
    backend/.venv/bin/python -B -m scripts.audio_matched_background_score_v250 "$phase" "$batch" || exit 1
  done
done
backend/.venv/bin/python -B -m scripts.audio_matched_background_score_v250 readout
```
