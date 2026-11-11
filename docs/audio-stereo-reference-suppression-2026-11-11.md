# Two-microphone noise suppression before speech admission

The earlier stereo selector recovered none of the mono rejections in Surrey or BBC. This comparison estimates a noise reference from the difference between the two microphones. Speech near the midpoint should be similar in both channels, while sound from one side can survive the difference. Reverberation and uneven microphones can put some speech in the reference as well. [Prior Surrey comparison](audio-measured-surrey-activity-transfer-2026-11-11.md) · [Two-channel cancellation and speech leakage](https://doi.org/10.3390/app11062816)

## Fixed comparison

Use the exact prior Surrey ±60° renders and the matched BBC holdout renders. The two Surrey roles share no foreground group, declared voice, decoded foreground, or background source. Each has two genuine, two expressive synthetic, and two stock synthetic speech groups. BBC repeats the Surrey holdout foregrounds and noise sources in a different measured room. All three partitions contain six gain-matched clean and 12 noisy speech inputs. Joint stereo SNR for noisy cases is approximately −10 dB. The foregrounds and noise recordings were used in the earlier studies. [Room geometry and provenance](audio-measured-surrey-activity-transfer-2026-11-11.md)

Each noisy stereo input yields one control by subtracting the exact same-gain clean channels in signed sample coordinates. The residual must match the previous background hash. The same production decoder converts the controls to mono on the original three-second sample clock. There are 12 background-only controls per partition. They share the two noise recordings within each role, and no independent listening labels were added.

The two fixed candidates fit a frequency-domain coefficient from the production mono input and half the left-minus-right difference. Both use 512-sample Hann frames stepped 128 samples. The first fits all complete frames and the second uses the lowest-energy third of complete frames. Each divides the cross-power by the reference power plus 0.1 times the primary power and a numerical floor. Both cap coefficient magnitude at 1.5, limit subtraction to 150–6,000 Hz, and apply 90% of the estimated contribution. The output retains the exact sample count, with no loudness normalization. Both candidates were locked before the two separate role runs. [Candidate implementation](../backend/scripts/audio_stereo_reference_suppression.py)

Each view passes through the unchanged speech-activity checkpoint and full-profile three-second window rule. The activity model's returned ranges must cover at least half the input. The conditional candidate keeps every original admission and admits a rejected input only if the conditioned view meets that same rule. The candidate observes the mixed stereo recording alone. Known separate components are used only to audit SNR and speech loss after the decision.

## Local results

All 90 cases and 25,380 speech-activity forwards completed. The CPU runs took 4.396 and 8.827 seconds after model setup, with peak process RSS of about 549 MB. All 54 unconditioned mono activity spans reproduced the earlier results. An independent calculation checked the support intervals and admission arithmetic for all 270 views. No candidate clipped an output sample. [Exact aggregate](evaluation-panels/audio-stereo-reference-suppression-2026-11-11.json)

| Partition and input | Production mono | Mono or whole-clip candidate | Mono or quiet-third candidate |
| --- | ---: | ---: | ---: |
| Surrey development clean speech | 3/6 | 3/6 | 3/6 |
| Surrey development noisy speech | 0/12 | 0/12 | 0/12 |
| Surrey holdout clean speech | 4/6 | 4/6 | 4/6 |
| Surrey holdout noisy speech | 4/12 | 5/12 | 4/12 |
| BBC room transfer clean speech | 4/6 | 5/6 | 5/6 |
| BBC room transfer noisy speech | 8/12 | 8/12 | 8/12 |
| All matched background-only controls | 0/36 | 0/36 | 0/36 |

The whole-clip method recovered one noisy genuine Surrey view. Both arms recovered the same clean BBC view. Neither arm added a noisy synthetic view. Whole-clip filtering increased median SNR by 1.11 dB in Surrey development, 0.81 dB in Surrey holdout, and 1.11 dB in BBC. Quiet-third filtering produced smaller gains on the holdout rooms, including a 0.25 dB median loss in Surrey holdout. The source and background component replays reconstructed the conditioned mixtures to within one decoded sample unit, apart from floating-point rounding.

This experiment measures admission under simulated stereo mixtures and selected measured impulse responses. It ran zero authenticity forwards. The limited gain on noisy speech and the lack of recovered noisy synthetic views leave the method experimental. In the earlier −10 dB traffic trial, all three speech-gate rescues remained inconclusive after authenticity scoring. [Earlier downstream authenticity test](audio-noise-speech-rescue-downstream-v270-result.md)

## What calibration alone could recover

A separate replay opened the exact saved raw authenticity responses from three prior studies. It measured pairwise ROC AUC and counted synthetic responses above the highest genuine response in each condition. That cutoff yields zero observed genuine flags in the same cohort by construction. A second retrospective calculation allows one genuine flag. These counts bound the possible separation on each archive without fitting a calibration, changing the operating rule, or making new model calls. [Exact replay aggregate](evaluation-panels/audio-noise-rank-replay-2026-11-11.json)

| Archived condition | Scored genuine / synthetic | Raw ROC AUC | Recorded synthetic AI flags | Synthetic above all genuine scores |
| --- | ---: | ---: | ---: | ---: |
| v270 matched clean | 8 / 15 | 1.000 | 14/15 | 15/15 |
| v270 traffic −10 dB | 8 / 15 | 0.733 | 0/15 | 4/15 |
| v219 traffic 0 dB, first generator | 11 / 16 | 0.744 | 0/16 | 1/16 |
| v219 traffic 0 dB, second generator | 11 / 12 | 0.947 | 7/12 | 8/12 |
| v208 river 0 dB | 18 / 24 | 0.838 | 0/24 | 4/24 |

At −10 dB traffic, allowing one genuine flag raises the same-cohort maximum to 6/15 synthetic. Both archives with 0 dB noise also show reduced ranking or a generator-specific loss. This separates two effects. The existing point misses some synthetic clips whose raw score still exceeds the genuine scores, and stronger noise also removes score separation that a monotone calibration cannot restore. A separate detector or a different representation remains important for the hardest conditions. These historical replays use assessed clips and small, condition-specific groups. The −10 dB cohort is a selected subset of the earlier fresh-noise foregrounds. [Traffic −10 dB result](audio-noise-speech-rescue-downstream-v270-result.md) · [Shared-noise result](audio-fresh-shared-noise-score-v219-result.md) · [River-noise result](audio-demand-transfer-v208-result.md)
