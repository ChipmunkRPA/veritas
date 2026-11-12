# Direct short-window voice scoring on original audio

On six new edits under competing room speech, none of the three comparisons produced a useful location at its frozen point or at up to two matched false genuine groups. The 1.20-second scoring path cleared the current check's one fixed false genuine group. It did not improve edit localization. [Aggregate results](evaluation-panels/audio-short-native-response-transfer-2026-11-12.json)

## Frozen test

The existing speech detector scored native 1.20-second samples with a 0.60-second hop. The comparison used the same checkpoint on its existing experimental two-second windows starting at 1 and 3 seconds. Both routes received the original mono PCM and the ordinary half-window speech admission rule. The existing model's usual input preprocessing and zero padding still apply. A reported interval covers only original source samples. This test changes the amount of speech presented to the classifier and does not use the previous splice boundary cue, separated audio, or a speaker reference. [Prior boundary study](audio-variable-boundary-transfer-2026-11-12.md) · [Existing checkpoint terms](commercial-model-audit.md)

A third arm combines the shorter score with the highest overlapping two-second score when at least 0.60 seconds of native time is shared. Separate medians and interquartile ranges from the genuine point controls standardize the scores before averaging. The combined arm retains the selected 1.20-second interval. All three fixed points come from the strongest file response in 12 previously used genuine selection controls. The two-second point responses were replayed from exact earlier native-window commitments. Short point responses were measured before any held inference. A useful interval covers at least half the known edit and reaches 0.30 temporal intersection over union. Equal-false-group results compare retrospective ranking at the same false genuine group budgets.

The held test uses six new generated inserts across two previously qualified synthesis families, six same-reader ordinary inserts, six other-reader ordinary inserts, and two isolated-room controls. There are six distinct host readers, six separate ordinary donor readers, and six generated voice profiles. The retained generated sources were matched to exact publisher commitments and screened against earlier raw bytes, scripts and decoded samples. The original voice banks and profiles have earlier project exposure. The exact inserts and original submitted mixtures are new. The genuine clips use exact source samples after the earlier analyzed four-second prefixes. [Genuine source terms](https://www.openslr.org/12/) · [Expressive archive terms](https://huggingface.co/datasets/laion/elevenlabs-v4-five-language-voice-acting) · [Qualified stock source terms](audio-short-edit-source-plan-v169.md)

Both held AMI rooms have a separate participant cohort. They were used earlier in other studies, and the new exact recording intervals exclude the recorded prior cuts. Each room contains edits of 0.64, 0.80 and 0.96 seconds and includes both generators. At least two annotated competing voices overlap every held edit for 510–960 ms. Whole-clip foreground to background ratio is −5 dB. Within each group the three matched waveforms share one background and the same PCM outside the exact edit. [AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Held results

| On six AI edits and 14 genuine views | Two-second current | 1.20-second short | Cross-scale |
| --- | ---: | ---: | ---: |
| AI flags at fixed point | 0 | 0 | 0 |
| False genuine groups at fixed point | 1 | 0 | 0 |
| Useful AI locations at 0 / 1 / 2 equal false groups | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| AI flags at two equal false groups | 1 | 2 | 2 |
| Useful ranges before applying a point | 2 | 1 | 1 |
| Unassessed held files | 0 | 0 | 0 |

The current check's fixed false group includes both ordinary splices of the same source episode. Neither isolated-room recording flagged in the frozen or retrospective comparisons. Before speech admission or scoring, the candidate geometry could locate four of six edits at two seconds and all six at 1.20 seconds. The shorter candidate's extra flags at the two-false-group point did not cover the known edits. Within each family considered on its own, relaxing to two false groups flagged a useful expressive edit under all three arms and a useful stock edit only under the current check. The held sample is small and those family-specific budgets are retrospective.

## Replay

The local model ran 106 admitted short-window forwards during point selection, then 40 current and 179 short-window forwards on the 20 held files. The native audit regenerated the held mixtures and all reported outcomes. The independent reader verified the exact PCM and component sums, matched samples outside edits, native window hashes for both lengths, and the fixed and equal-false-group arithmetic. Neither audit reran neural inference.

```sh
base=backend/data/independent-evaluation
run="$base/audio-short-native-response-2026-11-12"
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_short_native_response_transfer audit "$run"
python3 -B -S backend/scripts/review_audio_short_native_response_transfer.py "$run" \
  "$base/audio-original-pcm-boundary-2026-11-12" \
  "$base/audio-same-speaker-enrollment-2026-11-12" \
  "$base/spectra-recovery-2026-11-11/ami/ami_public_manual_1.6.2.zip"
```

Use a new directory and execute `freeze`, `point`, `held`, then `report` to reconstruct a separate local receipt. [Runner](../backend/scripts/audio_short_native_response_transfer.py) · [Independent reviewer](../backend/scripts/review_audio_short_native_response_transfer.py)
