# Competing speech creates false AI flags on genuine audio

The guarded CPU run completed all 98 inputs and 84 clean-to-noisy
comparisons. The speech gate allowed 89 raw predictions and serving
requested 20 stability forwards. The worker completed in 69.15 seconds
with 2.76 GiB maximum sampled process-tree RSS and no guard or sampling
errors.
[Frozen method](audio-pacc-false-alarm-v177-method.md) · [Verified aggregate](evaluation-panels/audio-pacc-false-alarm-result-v177.json)

| Treatment | VCTK false AI | VCTK original | VCTK inconclusive | AMI false AI | AMI original | AMI inconclusive | AMI unavailable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 0/10 | 10 | 0 | 0/4 | 2 | 0 | 2 |
| Babble, 0 dB | 7/10 | 0 | 3 | 0/4 | 0 | 4 | 0 |
| Babble, +5 dB | 4/10 | 0 | 6 | 0/4 | 0 | 3 | 1 |
| Environmental noise, 0 dB | 0/10 | 1 | 9 | 0/4 | 0 | 3 | 1 |
| Environmental noise, +5 dB | 0/10 | 3 | 7 | 0/4 | 0 | 3 | 1 |
| Instrumental music, 0 dB | 1/10 | 0 | 9 | 0/4 | 0 | 2 | 2 |
| Instrumental music, +5 dB | 0/10 | 3 | 7 | 0/4 | 1 | 1 | 2 |

VCTK's seven 0 dB babble false flags span four of five utterances. The
four +5 dB flags span three. VCTK uses two microphones per utterance and
its speakers appear in the historical catalogue. The four AMI meeting
excerpts returned no false AI flags, with 16 inconclusive and nine
unavailable results across all 28 AMI variants.

Noise type matters in this sample. Competing speech drove AI flags on
the VCTK inputs, while environmental noise mainly reduced usable
conclusions. The nominal SNR applies to the full publisher clips.
Only genuine foregrounds were evaluated, so the effect of an
overlapping-speaker check on AI recall needs a separate matched study.
This run changes no serving rule or confidence calibration.
[Source qualification](audio-pacc-sample-v176-result.md) · [Publisher datacard](https://moonscapesoftware.ca/datasets/pacc-p/)
