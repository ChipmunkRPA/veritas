# Short speech edits measured with linear and Mel cepstral features

None of the three checks located a held short AI voice edit under loud competing speech at its frozen point. The combined features located one dry edit, while the incumbent located two. The experiment did not qualify for the app. [Results and exact sample commitments](evaluation-panels/audio-cepstral-partial-transfer-2026-11-12.json)

## Frozen comparison

Linear and Mel frequency filterbanks give different emphasis to high frequencies and speech formants. A recent study combines these features and reports very different localization and classification performance across corpora. This experiment implements both filterbanks directly, tests them independently and together, and uses no external detector code or weights. [Cross-corpus cepstral study](https://arxiv.org/abs/2605.29531v3) · [Component-specific spectral analysis](https://arxiv.org/abs/2610.17617)

On the original 16 kHz mono waveform, form 25 ms Hann windows every 10 ms inside each 200 ms cell. Use 32 triangular frequency bands from 60 to 7,800 Hz, log band power, then cosine coefficients 1 through 12. The 48 features per view combine mean, dispersion, absolute adjacent frame differences, and a contrast with adjacent cells. The combination uses both 48-feature blocks. Skip a cell below the fixed energy floor. Use the highest average over three consecutive cells with at least two measured cells, and retain that 600 ms span on the original clock.

The three small ridge heads use identical matched splices and genuine controls from eight fitting hosts, with exact overlap fractions as targets. All three points use the strict maximum of the same 46 genuine point-selection files. The four-second and six-second held recordings use different source groups, and the six-second panel uses recorded AMI room speech. All files had prior project exposure. No held input entered fitting or point selection. Useful means at least 0.30 interval IoU and half the edit covered. [Source roles and licenses](audio-periodic-excitation-transfer-2026-11-12.md#method-and-source-roles)

| Held condition | Current | Linear | Mel | Combined |
| --- | ---: | ---: | ---: | ---: |
| Dry | 2/10 | 0/10 | 0/10 | 1/10 |
| Competing speech, −5 dB | 0/10 | 0/10 | 0/10 | 0/10 |
| Competing speech, +15 dB | 1/10 | 0/10 | 0/10 | 0/10 |
| AMI-panel dry | 2/6 | 0/6 | 0/6 | 0/6 |
| AMI room speech | 0/6 | 0/6 | 0/6 | 0/6 |

All held files produced a measurable span. The linear and combined arms flagged no genuine held files. The Mel arm falsely flagged one dry, one +15 dB and one AMI dry delivery. None of the arms flagged isolated background. At a retrospective allowance of one genuine source group falsely flagged in the −5 dB panel, the combined arm found one useful edit, while the incumbent found two. The strongest combined range overlapped usefully on two of ten −5 dB edits before applying a point, and on zero of six room mixtures.

## Replay

```sh
python3 -B -IS backend/scripts/review_audio_cepstral_partial_transfer.py \
  docs/evaluation-panels/audio-cepstral-partial-transfer-2026-11-12.json
```

The public reader recomputes the reported spans for all 216 cases, the decision points, and the cell counts from the recorded scores. For native replay, use the exact [parent and meeting inputs](audio-periodic-excitation-transfer-2026-11-12.md#verification). The native replay verifies their sample commitments and recomputes every feature, ridge head, point and decision. Raw audio stays local.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_cepstral_partial_transfer run "$run" --parent "$parent" --meeting "$meeting"
backend/.venv/bin/python -B -m scripts.audio_cepstral_partial_transfer review "$run" --parent "$parent" --meeting "$meeting"
```
