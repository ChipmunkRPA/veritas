# A selective second opinion for synthetic speech under competing voices

The frozen paired correction found 10 additional synthetic speech mixtures and added eight genuine false flags. The selective second opinion reproduced the correction because there were no withdrawn incumbent flags to restore. The earlier false-flag reduction did not transfer to these rooms. The current audio check stays in service. [Aggregate and independent replay](evaluation-panels/audio-disagreement-transfer-2026-11-12.json)

## Fixed comparison

Compare the current voice classifier, its previously fitted paired acoustic correction, a separately calibrated independent classifier, and a policy that asks the independent classifier to resolve one type of disagreement. The two classifiers analyze the same admitted three-second mono window. The paired correction uses the current classifier's pooled speech representation. The separate classifier is the pinned Apache-2.0 Spectra-AASIST3 CPU INT8 release. It has a different encoder and decision head. No weights were fitted or downloaded for this comparison. Recent benchmarks report selective combination of distinct audio classifiers under acoustic shift. [Paired correction](audio-paired-nuisance-transfer-2026-11-11.md) · [Independent checkpoint and terms](audio-independent-background-transfer-2026-11-11.md) · [AT-ADD](https://arxiv.org/abs/2608.14249) · [ESDD2](https://arxiv.org/abs/2606.10791)

The standalone second classifier must exceed its maximum assessed genuine or isolated-background response in the separate calibration cohort. The combined policy can restore an incumbent AI flag withdrawn by the paired correction. It uses the highest second-classifier response among genuine and isolated controls with that exact disagreement. If calibration produces no such cases, the policy falls back to the standalone operating point. The two candidate points use strict excess. Original and unassessed results keep their existing direction. The experiment uses internal operating points and supplies no customer confidence probabilities.

## Source separation

Thirty distinct English foreground groups come from the previously qualified LibriSpeech, DuplexGen PLN, ElevenLabs archive, and Kokoro banks. Calibration uses seven genuine and six synthetic groups. Confirmation uses eight genuine and nine synthetic groups, with three groups per synthetic family. Recorded group, voice and exact source commitments are disjoint across calibration and confirmation. The nine held synthetic groups were excluded from the prior 77-view room transfer. The genuine readers and generator banks have earlier project exposure. Model training overlap and unrecorded identities remain unknown. [Source rights and lineage](audio-short-edit-source-plan-v169.md)

Use one AMI meeting for calibration and two different meetings for confirmation. For each meeting, manually annotated words identify two new three-second cuts with two and three competing speakers. The simultaneous word durations across the six cuts range from 490 to 1,390 ms. These meetings have prior speaker-count evaluation. The six source windows are outside that study's selected time ranges. Each foreground is rendered at −5 dB foreground-to-background ratio in all cuts assigned to its role and supplied as a gain-matched clean control. The six background cuts also appear on their own. This produces 41 calibration and 89 confirmation views. Repeated mixtures share their foreground and room source, and the two voices conditions describe the background alone. [Native source lineage](audio-stream-presence-transfer-2026-11-11.md) · [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Local results

Calibration assessed 21 genuine foreground or isolated-background views and produced no controls where the correction withdrew an incumbent AI flag. The policy therefore used the standalone point, as specified before measurement. This calibration supplied no separate estimate of false alarms among disputed flags.

| Held scene | Current true / false AI flags | Paired correction true / false | Independent true / false | Selective second opinion true / false |
| --- | ---: | ---: | ---: | ---: |
| Clean, 9 synthetic and 8 genuine | 6 / 0 | 7 / 0 | 6 / 0 | 7 / 0 |
| Two competing voices, 18 synthetic and 16 genuine | 7 / 3 | 12 / 6 | 3 / 1 | 12 / 6 |
| Three competing voices, 18 synthetic and 16 genuine | 9 / 2 | 14 / 7 | 2 / 0 | 14 / 7 |

The correction added four cloned, five expressive, and one stock synthetic flag across the held mixtures. It introduced one additional false flag in the first room and seven in the second. Three of the latter appeared under two competing voices and four under three competing voices. Each room contributes eight genuine readers per condition. All mixed foregrounds were assessed. One clean synthetic clip and one isolated background were unassessed. The other three isolated backgrounds stayed inconclusive. One synthetic mixture with three competing voices kept its incumbent original direction under every arm.

The standalone classifier reduced false AI flags from five to one across the noisy scenes, with correct synthetic flags falling from 16 to five. It also introduced one false flag on a different genuine view while clearing the five current false flags. The selective rule addressed a specific failure mode from the previous study. Neither this calibration nor the held panel produced an incumbent flag withdrawn by the correction, so this experiment could not measure whether the second opinion restores that flag safely. Three of six frozen advancement criteria passed.

The four CPU scoring stages made 142 incumbent classifier forwards, including current stability probes, and 126 independent classifier forwards. The first model peaked at 2.97 GB RSS and the second at 0.96 GB. The stages took about 179 seconds in total. The independent replay reconstructed all 130 PCM deliveries and 126 independent classifier input hashes, and reproduced 32 aggregate decision cells and the advancement result from saved scores. It did not repeat model inference. Raw audio, per-clip scores, voice identities and the exact room windows stay local.

## Reproduce locally

Use the completed paired correction, the pinned AMI annotation ZIP, the three archived first-microphone AMI recordings, the retained Spectra files, and the existing serving audio assets. The runner also checks the retained source receipts and expects an ONNX Runtime installation in the backend environment. Use a new local directory for each run.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
args=("$run" "$paired" "$annotations" "$rooms" "$assets" "$weights")
backend/.venv/bin/python -B -m scripts.audio_disagreement_transfer freeze "${args[@]}"
for panel in calibration confirmation; do
  for phase in voice secondary; do
    backend/.venv/bin/python -B -m scripts.audio_disagreement_transfer "$phase" "${args[@]}" --panel "$panel" || exit 1
  done
  if [ "$panel" = calibration ]; then
    backend/.venv/bin/python -B -m scripts.audio_disagreement_transfer fit "${args[@]}" || exit 1
  fi
done
backend/.venv/bin/python -B -m scripts.audio_disagreement_transfer report "${args[@]}"
backend/.venv/bin/python -B -m scripts.review_audio_disagreement_transfer "${args[@]}"
```
