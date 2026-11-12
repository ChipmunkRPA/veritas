# Paired correction for voice screening under competing speech

A linear correction trained on matched clean and mixed speech reduced genuine false flags in both recorded AMI scenes. It moved one correct synthetic flag from the overlapping-voices scene to the mostly-single-voice scene. The separate four-second selection role added a false flag under equal-level competing speech, and the held gains came from one synthesis family. The frozen advancement criteria failed. [Aggregate and replay receipts](evaluation-panels/audio-paired-nuisance-transfer-2026-11-11.json)

## Frozen comparison

The current voice detector outputs a raw log-odds score from a linear head over its pooled speech embedding. The fit uses differences between embeddings of the same foreground before and after the addition of competing speech. These differences capture a measured response to that acoustic change. Their source labels play no part in fitting the correction. This differs from the earlier label-trained, paired consistency heads. The comparison keeps the installed voice weights and adds no encoder, speech separation, or second waveform at inference. [Prior paired heads](audio-cross-layer-room-transfer-2026-11-11.md) · [Acoustic robustness benchmark](https://arxiv.org/abs/2608.23437v2) · [Checkpoint terms](commercial-model-audit.md)

Fit one correction vector by minimizing its squared effect on centered clean embeddings plus four times the squared remaining score drift across the paired clean and competing-speech embeddings. Add a ridge equal to one quarter of the mean clean feature variance. Give each available foreground equal weight and weight its admitted speech mixtures equally within that foreground. The correction subtracts a centered linear term from the incumbent's raw log odds. The fit used 110 clean anchors and 330 paired shifts. On fitting pairs, root-mean-square logit drift fell from 4.42 to 1.42, with a 2.82 RMS change on the clean anchors.

Select one fixed internal point each for raw score and corrected score on the separate calibration role. Both points use the same rule. They permit at most the incumbent genuine false flags in each of the nine calibration conditions. At inference, the same point applies to all assessed inputs without a condition label. The diagnostic rule can change an AI flag to inconclusive or raise inconclusive to AI. It preserves original and unassessed directions. These are benchmark decisions with no probability interpretation.

The archived four-second panel has four disjoint roles, each with 116 foreground recordings, 12 recorded backgrounds, and 1,044 planned views. Exact foreground and background commitments, declared connected groups, and view PCM commitments are disjoint across roles. All four roles were used by earlier project experiments. Synthesis families are shared across roles. Pretraining overlap and unknown aliases remain unverified. [Archived source roles and delivery conditions](audio-combined-confidence-v84-final.md)

## Four-second results

Each row has 60 synthetic and 56 genuine planned views. The score-only comparator matched the incumbent flag counts in these cells. The held study also records all other conditions, original directions, inconclusive results and unassessed views.

| Role and condition | Synthetic flags, current → corrected | Genuine false flags, current → corrected | Assessed ROC AUC, raw → corrected |
| --- | ---: | ---: | ---: |
| Selection, clean | 55 → 55 | 0 → 0 | 1.000 → 1.000 |
| Selection, competing speech 0 dB | 52 → 51 | 9 → 10 | 0.929 → 0.935 |
| Selection, competing speech +5 dB | 59 → 59 | 15 → 11 | 0.966 → 0.979 |
| Confirmation, clean | 55 → 56 | 0 → 0 | 1.000 → 1.000 |
| Confirmation, competing speech 0 dB | 46 → 46 | 2 → 0 | 0.946 → 0.979 |
| Confirmation, competing speech +5 dB | 50 → 54 | 4 → 1 | 0.951 → 0.980 |

At confirmation 0 dB, the equal synthetic totals conceal three additions and three losses. At +5 dB there were four additions and no losses. All seven added synthetic flags across these two held cells came from expressive synthetic voices. Confirmation therefore failed the two-family requirement. The separate selection role failed the net-detection and false-alarm requirements.

The study reuses the original checkpoint responses. The independent readback bound all 48 archive chunks, verified 4,176 case bindings, reconstructed 3,132 decision records and 54 rank cells, reproduced the two operating points and verified the fit's normal equation directly from sample gradients. The first local integrity run stopped before fitting because skipped inputs have a separate published current status. The completed replay uses that status and retains skipped clips as unassessed.

## Native three-second AMI transfer

Keep the four-second fit and points fixed. Test 77 exact three-second views from the earlier separately qualified AMI panel. That panel has eight genuine reader groups, nine synthetic voice groups across three generator families, and two previously recorded rooms. No recorded foreground hash or group matches the four-second panel. The same foregrounds repeat across conditions, and the two room recordings have earlier project exposure. The two background-only cuts and the clean recordings serve as controls. [Exact sources and permissions](audio-independent-background-transfer-2026-11-11.md)

| Condition | Synthetic flags, current → corrected | Genuine false flags, current → corrected | Assessed ROC AUC, raw → corrected |
| --- | ---: | ---: | ---: |
| Clean | 7/9 → 7/9 | 0/16 → 0/16 | 1.000 → 1.000 |
| Mostly one competing voice, −5 dB | 4/9 → 5/9 | 3/16 → 1/16 | 0.639 → 0.757 |
| Overlapping competing voices, −5 dB | 5/9 → 4/9 | 12/16 → 7/16 | 0.438 → 0.563 |

The corrected score cleared seven genuine false flags across the two noisy scenes, with nine synthetic flags in total for both rules. It recovered one synthetic recording in the mostly-single-voice scene and lost one in the overlapping-voices scene. The two clean synthetic clips without a result stayed unassessed. Neither background-only control acquired an AI flag. The score-only comparator reproduced the incumbent flag counts on this panel.

The CPU run made 75 forwards on the existing admitted windows, took 51.1 seconds and peaked at about 2.91 GB RSS. Its exact head reproduced the saved raw responses within 8.15 × 10⁻⁸. The no-inference reviewer also rehashed all 77 native waveforms and recomputed rank and outcome counts from the scalar ledger. These findings apply to the tested source groups and rooms. The correction remains experimental.

## Reproduce locally

Use the exact v84 full archive, its retained four-role variant manifest and original head receipt, the separately qualified AMI comparison directory, and the installed checkpoint. Run from the repository root with new output directories. The public artifact contains aggregate counts and commitments. Waveforms, voice representations, fitted vectors and case-level responses remain local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_paired_nuisance_transfer freeze "$study" "$archive"
backend/.venv/bin/python -B -m scripts.audio_paired_nuisance_transfer run "$study" "$archive"
backend/.venv/bin/python -B -m scripts.audio_paired_nuisance_native freeze "$study" "$ami" "$native"
backend/.venv/bin/python -B -m scripts.audio_paired_nuisance_native score "$study" "$ami" "$native"
backend/.venv/bin/python -B -m scripts.review_audio_paired_nuisance_transfer "$study" "$archive" "$review" \
  --native-run "$native" --native-source "$ami"
```
