# Learned spectrogram checks for short voice edits

Two small spectrogram models and a combination with the current voice check missed all ten short edits under loud competing speech at their frozen operating points. At one or two false genuine groups, the current check located two of those edits and the candidates located zero. The candidates stay experimental. [Aggregate results and commitments](evaluation-panels/audio-local-spectrogram-transfer-2026-11-12.json)

## Method and training data

Train two small models from scratch. The first uses 200 ms local log-mel spectrogram features and their time differences. The second also reads the two nearest separated cells on each side and compares their representations with the local cell. Both models learn short time-frequency patterns through small temporal convolutions. A paired objective asks a synthetic insert to score above two matched ordinary splices inside the known edit, while discouraging score differences far from the edit. The approach uses no externally downloaded weights or code. Recent work on partial speech detection motivates training a localized model and measuring false alarms on benign edits separately. [Cross-corpus partial speech study](https://arxiv.org/abs/2605.29531) · [Benign edits and temporal anchors](https://arxiv.org/abs/2610.04479) · [Implementation](../backend/scripts/audio_local_spectrogram_transfer.py)

Construct 33 new training episodes from the previous role-qualified local source bank. Seven clone and four stock donor groups each contribute three episodes. Every episode includes an ordinary same-host splice, an ordinary different-voice splice and an AI insert at the same boundaries, with matched fades and gains. All three variants appear dry and with the same louder competing speech for 198 training inputs. Exact source, speaker and background groups remain separate from the operating-point and held roles under recorded ancestry. The source bank had previous project exposure. No expressive examples enter fitting or point selection. The new model scores the independent meeting recordings and held views after freezing the fit. [Qualified source roles and rights](audio-partial-residual-transfer-2026-11-12.md) · [LibriSpeech terms](https://www.openslr.org/12/) · [AMI Corpus license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Compare the local and contextual models separately. The combination averages standardized scores from the contextual and existing frame checks. Estimate centers and scales on 46 genuine operating-point clips. Each new arm uses strict excess of its highest genuine selection score, while the current point is unchanged. The strongest three consecutive cells give a 0.6-second range. A useful location needs 0.30 intersection over union and 0.50 edit coverage. A descriptive sweep separately compares rankings at identical numbers of falsely flagged genuine source groups. On the six-second AMI panel, compare the two trained checks without refitting. Use the earlier 0.6-second frame results for fixed-point comparison and the separately retained two-second scores for equal-alarm ranking. The prior AMI frame replay did not retain cell arrays for the combination. [AMI panel and baseline](audio-overlap-window-transfer-2026-11-12.md)

## Held results

| Condition | Current useful / genuine false | Local useful / genuine false | Context useful / genuine false | Combined useful / genuine false |
| --- | ---: | ---: | ---: | ---: |
| Dry, 10 edits and 20 genuine | 2 / 0 | 0 / 2 | 0 / 2 | 1 / 2 |
| Competing speech −5 dB, 10 edits and 34 genuine | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Competing speech +15 dB, 10 edits and 20 genuine | 1 / 0 | 0 / 2 | 0 / 2 | 0 / 1 |
| Clean AMI-panel hosts, 6 edits and 6 genuine | 2 / 1 | 0 / 0 | 0 / 0 | — |
| Loud AMI room speech, 6 edits and 6 genuine | 0 / 0 | 0 / 0 | 0 / 0 | — |

In the strong LibriSpeech interference, the edited file ranked above both matched ordinary splices in 7 of 10 comparisons for the incumbent, 2 for each learned model and 6 for the combination. Before applying points, the strongest range overlapped usefully in five incumbent, four local and one contextual or combined case each. In the held AMI scenes, the local model scored the AI edit above its matched genuine splice in 5 of 6 pairs, compared with 2 of 6 for the earlier two-second check. At one identical false alarm they each located one useful edit. This comparison supports testing the ranking and location separately. The learned candidates did not increase useful locations at the same false-alarm count and add no serving evidence or confidence weight.

The equal-alarm AMI locations came from different clips. The local check placed a stock-voice edit at 3.600–4.200 seconds against known support at 3.330–4.130 seconds. The two-second check placed another stock-voice edit at 3.000–5.000 seconds against known support at 3.080–3.880 seconds. A retrospective union of the local check at one false group and the two-second check at two covered both clips with two total false groups. Those cuts use the held controls, so the complementary locations support an independent trial rather than a calibrated combined rule.

## Replay

The report stage evaluated 216 files in 1.12 seconds after training. The four-second current comparator reproduced every archived score, range and flag exactly. A second fit reproduced both checkpoint and point files byte for byte. The runner regenerated the spectrogram inputs and responses. A separate standard-library reviewer rechecked the 216 exact PCM inputs, fixed and equal-alarm decisions, source-clock locations and the retrospective AMI union. A signal-only setup pass found a clipping component before model fitting. The frozen source builder uses its fixed host order to advance past unusable source spans and clipping. The aggregate contains only counts and commitments. Audio and learned weights stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze fit report audit; do
  backend/.venv/bin/python -B -m scripts.audio_local_spectrogram_transfer "$phase" "$run"
done
python3 -I -S backend/scripts/review_audio_local_spectrogram_transfer.py "$run" "$parent" "$meeting"
```

The runner accepts `--parent` and `--meeting` overrides for the retained four-second partial-residual and final six-second overlap-window directories.
