# Target-onset steering for voices under overlapping speech

The steered view recovered four synthetic clips at the two frozen operating points. Genuine false flags increased from one to three. At the same false-alarm count, the steered view detected fewer synthetic clips than mono at every nonzero count checked. The service continues to use its existing voice detector and calibration. [Aggregate results and commitments](evaluation-panels/audio-target-onset-transfer-2026-11-12.json)

## Fixed comparison

The earlier stereo experiment assumed the target arrived equally on both microphones. Delaying and attenuating one channel caused the filter to remove some target speech along with interference. This experiment derives the target direction from a supplied half-second interval in which the target is audible alone. The target cue uses timing and level between microphones. It uses no speaker embedding or enrollment database. Target-conditioned speech extraction and frequency-domain distortionless beamforming motivate the separation of the target cue from the interval assessed for authenticity. [Earlier fixed-center experiment](audio-spatial-interference-2026-11-11.md) · [Removed-source leakage](audio-removed-source-transfer-2026-11-12.md) · [Gannot, Burshtein and Weinstein](https://www.eng.biu.ac.il/~gannot/articles/tfgsc.pdf) · [Google VoiceFilter research](https://google.github.io/speaker-id/publications/VoiceFilter/)

Split each exact three-second foreground into the half-second onset and the remaining 2.5-second query. Introduce a fixed delay of 1–7 samples and attenuation of 1.5–4.5 dB in one channel. Mix the query with the retained simultaneous-speech part of the exact measured stereo background at −5 dB relative to the mono target. The query starts at sample 8,000 and stops at sample 48,000 on the synthetic scene clock. Recover a lag and gain from the two onset channels when both have RMS at least −45 dBFS and their lag-aligned correlation reaches 0.95. Render a distortionless two-channel projection using a 512-sample Hann transform and a 128-sample hop. Estimate covariance outside each half-second analysis block with diagonal loading. Use the normalized steering vector itself below 150 Hz and above 6,500 Hz. Its two channel weights depend on measured gain. The frozen runner’s phrase “equally weighted” refers imprecisely to this out-of-band operation. The numerical projection always uses the recorded steering-vector formula. [Implementation](../backend/scripts/audio_target_onset_transfer.py)

Score the mono and projected views with the same pinned local classifier and production Full speech admission. Keep mono when the onset is unqualified or the steered query fails admission. For each arm, choose the stricter of the maximum clean or native genuine response and the third-highest noisy genuine response in selection. Strict excess flags the result. Compare the two views at their frozen points first, then retrospectively compare ranking at exactly equal held genuine false counts. Also record the current serving direction on the same mono input. The frozen points and retrospective cuts are experimental operating points. [Classifier and activity asset audit](commercial-model-audit.md)

## Sources

Reuse the earlier source-disjoint stereo panel. The selection role contains 17 foreground groups, two measured AMI rooms, and 59 views. The held role contains seven genuine and nine synthetic foreground groups, two different rooms and 50 views. Clone, expressive and stock synthesis each contribute three held foregrounds. Every held foreground recurs across two noisy rooms, so 18 synthetic views represent nine foreground groups. Both roles contain clean foreground controls and original native room cuts on their unchanged mono track after the same clock trim. The development and held roles are disjoint by declared foreground group, exact foreground PCM, and room. The recordings and generator families had previous project exposure. The experiment uses newly rendered, fixed geometry and a known isolated onset. It does not establish performance on unknown speakers, moving sources, natural microphone paths, mono uploads, or short partial edits. The room material is CC BY 4.0. Foreground qualification retains the existing reader and generated-audio rights declarations. The projection uses no new model or external code. [Source qualification](audio-short-edit-source-plan-v169.md) · [LibriSpeech source terms](https://www.openslr.org/12/) · [AMI recording terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Local results

| Held noisy comparison | AI flags /18 | Genuine false flags /14 |
| --- | ---: | ---: |
| Current serving mono direction | 8 | 4 |
| Mono at its frozen experimental point | 0 | 1 |
| Onset-steered at its frozen experimental point | 4 | 3 |

All held noisy views were speech-admitted. Each room supplied five accepted synthetic and six accepted genuine onsets. Across accepted onsets, median SNR gains were +0.55 and +4.11 dB. Median target-component error was −47.4 and −44.7 dB relative to the target at the reference microphone. All four new synthetic flags were from distinct, actually steered clips, two clone and two expressive. The steered arm cleared one genuine false flag and introduced three. No genuine clean or native control was flagged by either experimental arm. Both arms flagged all nine clean synthetic controls. One of the two native held cuts was unassessed.

| Equal held false flags | Mono AI flags /18 | Onset-steered AI flags /18 |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 2 | 1 |
| 2 | 2 | 1 |
| 4 | 10 | 6 |

The same-false-alarm comparison showed no detection improvement from steering on this panel. The method passed three of five advancement checks. It failed the no-additional-false-alarm check and the requirement for at least 2 dB of acoustic gain in each room. The source conditions and accepted-onset counts limit the conclusion to this controlled panel. A separate short-edit study found that even near-ideal suppression can leave the model’s short-edit ranking unchanged. The two results suggest measuring target waveform recovery and authenticity discrimination separately. [Controlled attenuation on short edits](audio-room-reference-transfer-2026-11-12.md) · [Mixed-authenticity research](https://arxiv.org/abs/2609.03620)

## Reproduction

The first launch stopped on a detector method binding before classifier inference. The completed run used a fresh freeze after correcting the binding. It made 169 classifier forwards and 12,087 speech-activity frame forwards in 98.1 seconds, with approximately 2.96 GB peak process RSS. The source replay regenerated all 218 exact model-input PCM hashes. A separate standard-library reviewer verified the hashes and role disjointness, recalculated the points, held flags and room medians, and rejected a one-byte PCM change. Only aggregate outcomes and whole-artifact hashes are included in the result file. The source and scoring ledgers stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze selection fit confirmation report audit; do
  backend/.venv/bin/python -B -m scripts.audio_target_onset_transfer "$phase" "$run"
done
backend/.venv/bin/python -B -m scripts.review_audio_target_onset_transfer "$run"
```
