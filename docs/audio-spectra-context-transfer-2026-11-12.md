# Short voice edits across overlapping three-second contexts

The independent context scores found no held edits at the frozen points. An edge comparison located three of six AMI edits at two retrospective false alarms, compared with one for the earlier two-second check. The new result stays experimental. [Aggregate results and commitments](evaluation-panels/audio-spectra-context-transfer-2026-11-12.json)

## Method

Run the archived, commercially licensed CPU detector on fixed, contiguous three-second samples. The exact ONNX file and its base and derivative publisher cards match the earlier audit. The retained derivative and base cards declare Apache-2.0. Its quantization record describes the weight-only INT8 conversion. Model training overlap with the retained audio remains unknown. The local macOS arm64 ONNX Runtime 1.31.1 was reused directly from an existing cache. No new download or installation was needed. A repeated native smoke input produced identical graph bytes and output. [Prior weight qualification](audio-independent-background-transfer-2026-11-11.md) · [Archived derivative revision](https://huggingface.co/Limitless-8/spectra-aasist3-int8-audio-deepfake/tree/b56aed04853cb4e5bf825025c54c93d4bc345c61) · [Archived base revision](https://huggingface.co/lab260/Spectra-AASIST3/tree/bc0ded888080ddad493177bb53aa6f5b95219d7c)

Advance the start by one second. For each context, apply the published 0.97 pre-emphasis and repeat the prepared samples to 64,600 graph samples. The source location stays three seconds. Compare the highest raw context score with the largest score difference between adjacent contexts. A contrast proposes the one-second edge unique to the stronger window. The two shared seconds and interactions inside the classifier remain unresolved. Four-second files only have two outer candidate edges. Six-second files allow three adjacent comparisons. This geometry evaluates context sensitivity. It does not establish that the differing samples caused the response. [Temporal boundaries and genuine edits](https://arxiv.org/abs/2610.04479) · [Partial speech localization](https://arxiv.org/abs/2604.02913)

Use genuine −5 dB controls from the prior operating-point role to fix each new score at its genuine maximum. Compare the unchanged current check, both new views separately, and the union of the current and edge flags. Keep the two ranges if both flag. The previously tested four-second splices keep their distinct foreground and background roles, exact ordinary controls, fades and synthetic edit times. The six-second AMI clips supply a separate no-refit comparison with recorded competing speech. Both cohorts had earlier project exposure. Measure useful source location at 0.30 IoU and 0.50 edit coverage, and separately compare raw ranking at equal false source-group counts. A three-second range cannot meet the IoU condition for a 0.8-second edit, so detection flags are reported separately. [Qualified sources and earlier short edits](audio-partial-residual-transfer-2026-11-12.md) · [AMI case construction](audio-overlap-window-transfer-2026-11-12.md) · [AMI recording terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Held results

All fixed-point comparisons missed all ten known −5 dB matched edits and all six held AMI edits. The edge and union checks both falsely flagged the same isolated genuine background. The unchanged incumbent and the independent raw-score arm produced zero genuine false flags. Only three of the ten matched edits were geometrically reachable by either outer edge. Five of the six AMI edits had useful strongest-edge positions before applying a point, compared with three for the earlier two-second check and one for the earlier 0.6-second frame check.

| Held cohort and retrospective false groups | Earlier check useful | Independent raw score useful | Exclusive-edge contrast useful |
| --- | ---: | ---: | ---: |
| Matched −5 dB, 0 | 0 | 0 | 0 |
| Matched −5 dB, 1 | 2 | 0 | 0 |
| Matched −5 dB, 2 | 2 | 1 | 0 |
| AMI room speech, 0 | 0 | 0 | 0 |
| AMI room speech, 1 | 1 | 0 | 1 |
| AMI room speech, 2 | 1 | 0 | 3 |

The matched comparator is the retained 0.6-second incumbent frame score. The score-ranked AMI comparator is the retained two-second check. At two false groups, the AMI edge contrast nominated 3.000–4.000 seconds on a clone edit at 2.830–3.630 seconds, 2.000–3.000 on a clone edit at 2.080–2.880, and 3.000–4.000 on a stock edit at 3.080–3.880. The earlier check located the same stock edit. The AMI edge score ranked the edited clip above its matching genuine splice in four of six comparisons, compared with two for the earlier check. The false-alarm cuts in this table use held control scores, so the room result is a lead for a separate prospective comparison. It carries no serving weight.

## Replay

The CPU run made 180 model forwards across 22 point controls and 56 held files. The independent reviewer reconstructed each native three-second span and every prepared Float32 model input. It recomputed which one-second edge was exclusive to the stronger window, plus all fixed and equal-false-alarm counts. Use the archived model assets as `weights` and the retained four-second partial-residual and final six-second overlap-window directories as `parent` and `meeting`. An existing compatible CPU ONNX Runtime must be on the Python path.

```sh
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze selection confirmation report; do
  backend/.venv/bin/python -B -m scripts.audio_spectra_context_transfer "$phase" "$run" "$weights"
done
python3 -I -S backend/scripts/review_audio_spectra_context_transfer.py "$run" "$parent" "$meeting"
```
