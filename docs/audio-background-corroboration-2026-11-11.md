# Background-induced voice false alarms and corroboration

One of two loud backgrounds sharply increased the detector's score on genuine speech. Requiring the AI flag to hold after a bandpass removed two of five genuine false flags in the earlier panel while preserving all 25 synthetic detections. A separate challenge found no false flags for the rule to correct, so the current detection policy stays in place. [Earlier false-alarm comparison](audio-wavlm-two-background-transfer-2026-11-11.md) · [Exact aggregate](evaluation-panels/audio-background-corroboration-2026-11-11.json)

## What changed the score

Keep the same six English genuine recordings, clean counterpart, and exact isolated background samples. Recreate each background at 0 and +5 dB, and retain only its 100–3,800 Hz or remaining component at the original component level. Score each residual by itself at the matched level as well. This mechanism study runs the classifier directly on all 60 constructed inputs so speech admission does not censor the score comparison. Its initial directions describe the classifier response. In the earlier normal route, both isolated background recordings were unassessed.

| Known genuine sources or their isolated backgrounds | Background A | Background B |
| --- | ---: | ---: |
| Noise power in the 100–3,800 Hz band | 68% | 14% |
| Median raw score at −5 dB on earlier speech-admitted mixtures | 0.996 | 0.004 |
| Initial AI direction on isolated matched background controls | 6/6 | 0/6 |
| Initial AI direction on the genuine mixture at 0 dB | 2/6 | 0/6 |
| Initial AI direction on the genuine mixture at +5 dB | 2/6 | 0/6 |
| Original false flags reproduced using only the in-band background component | 3/4 | 1/1 |
| Original false flags reproduced with the in-band background component removed | 0/4 | 0/1 |

A different genuine input crossed into the initial AI direction for each background after the in-band component was removed. For background A, the higher scores follow the added background and persist under some quieter conditions. The in-band component carried more of its energy, so the split alone cannot isolate which frequencies the model responds to independently of level. The isolated noise checks repeat two underlying recordings at different matched gains. The simple content heuristic called the A residuals mostly music and the B residuals speech, although neither isolated source passed the serving speech gate. This disagreement makes the heuristic unsuitable as a sole authenticity check. [Existing content analysis method](../backend/app/detectors/audio/content.py)

## Does corroboration improve the decision?

Two candidate checks use the unchanged voice classifier and calibration. The first repeats the score with a 100–3,800 Hz bandpass. The second estimates a background spectrum from frames fully outside the original voice activity spans and tests a partially reduced view. It requires at least 25 eligible frames and never reduces a spectral bin below half amplitude. Each rule withholds an existing AI direction only when the transformed score loses the AI direction and its raw log odds fall by more than the fixed stability tolerance. It preserves unassessed cases. Both methods were frozen before the challenge inference. [Earlier spectral conditioning](audio-noise-conditioning-v211-result.md)

The challenge uses six genuine Turkish recordings, five designed synthetic voices, and three utterances from one stock synthetic voice. Each has clean, −5 dB kitchen, and −5 dB traffic views drawn from three distinct recordings per background category. This uses different foreground sources and background recordings from the English diagnosis. The Turkish foregrounds previously appeared in lower-noise baseline studies. These are new mixtures and policy measurements on source-disjoint roles. Earlier baseline exposure, unknown pretraining, and repeated sources limit independence. [RUEG source and rights](audio-rueg-genuine-v246-method.md) · [Alania source and rights](audio-alania-roster-v178-method.md) · [Okur source and rights](audio-okur-turkish-v240-method.md) · [Background provenance](audio-commons-background-v234-method.md)

| Challenge condition | Genuine falsely flagged / assessed | Synthetic flagged / assessed | Unassessed genuine / synthetic |
| --- | ---: | ---: | ---: |
| Clean | 0/5 | 8/8 | 1 / 0 |
| Kitchen, −5 dB | 0/3 | 1/8 | 3 / 0 |
| Traffic, −5 dB | No genuine input assessed | 0/8 | 6 / 0 |

Both rules gave the same decisions as the current check on all 42 views. There was no genuine false flag to correct in this challenge. The quiet-frame check could score just one of the nine existing AI flags, since the other eight lacked suitable quiet frames.

## Replay on the original failures

Apply the already frozen rules to the original English panel. The bandpass rule withheld one of four genuine false flags on background A and the sole genuine false flag on B. It preserved all 25 previously detected synthetic views across clean and noisy conditions. The quiet-frame rule changed no verdict, even though it had sufficient quiet frames for all four A false flags. It had insufficient quiet for the B false flag. The replay revisits the samples that motivated the checks. Its improvement supports a separate confirmation with more distinct in-band background sources and assessed genuine positives before any serving change.

The three local model stages completed 60 diagnostic forwards, 47 challenge forwards, and 52 retrospective forwards, with a maximum 2.91 GB process RSS. The challenge used 3,948 speech-activity frame forwards. Separate readouts verified the 102 prepared PCM hashes, 14 exact cross-role source exclusions, prior waveform commitments, and per-rule arithmetic. The aggregate leaves individual audio, scores and source identifiers local.

The noise-dose sweep follows the matched-mix strategy used in recent noise-robustness research. That research also reports noisy genuine speech being mistaken for synthetic speech. The local results point to testing a noise-aware representation with separate background recordings assigned to fit, calibration, and confirmation, since a simple score cutoff or bandpass can leave noise-related false flags. [Controlled-SNR audio detection study](https://arxiv.org/abs/2512.13744) · [Open-world calibration study](https://arxiv.org/abs/2611.11465)
