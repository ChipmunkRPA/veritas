# Background-noise detection varied sharply by voice generator

The local detector completed all five conditions on the fresh Turkish
panel. It scored all 195 prepared waveforms, with 262 classifier forwards
including stability checks. Every prepared clip passed the speech gate.
The five workers took 193.6 seconds in total and reached 2.77 GiB maximum
sampled worker-tree memory. [Method](audio-fresh-shared-noise-score-v219-method.md) · [Verified counts](evaluation-panels/audio-fresh-shared-noise-score-complete-v219.json)

| Background | Alania AI / original / inconclusive (16 clips) | Drifting AI / original / inconclusive (12 clips) | Genuine AI / original / inconclusive (11 clips) |
| --- | ---: | ---: | ---: |
| Matched clean | 15 / 0 / 1 | 12 / 0 / 0 | 0 / 11 / 0 |
| Kitchen, 0 dB | 3 / 4 / 9 | 12 / 0 / 0 | 0 / 11 / 0 |
| Kitchen, +5 dB | 5 / 4 / 7 | 12 / 0 / 0 | 0 / 11 / 0 |
| Traffic, 0 dB | 0 / 4 / 12 | 7 / 0 / 5 | 0 / 7 / 4 |
| Traffic, +5 dB | 2 / 5 / 9 | 11 / 0 / 1 | 0 / 9 / 2 |

At traffic 0 dB the detector missed all 16 Alania clips. Four received
wrong-original directions and twelve were inconclusive. It still flagged
seven of the 12 Drifting clips, with the other five inconclusive. At
kitchen 0 dB it flagged three Alania clips and all 12 Drifting clips.
None of the 11 prepared genuine clips received an AI direction. Four
were inconclusive at traffic 0 dB.

The denominator includes nine additional unavailable foregrounds in every
condition, five genuine contributors and four Drifting prompts. The 12
Drifting clips share three stock voices. The two generator cohorts use
different texts, voices and production histories. Each comparison against
matched clean audio uses the same clip, and each matched set shares the
same quantized background samples. [Source and preparation coverage](audio-fresh-shared-noise-v217-result.md)

The strongest candidate for further evaluation must recover AI flags on
the harder generator and noise combinations while tracking genuine false
AI and synthetic wrong-original directions separately. The earlier mild
conditioning experiment recovered zero AI flags on its river-noise
panel. A separate frozen comparison evaluates a second classifier on
those same river clips. [Conditioning result](audio-noise-conditioning-v211-result.md) · [Second-classifier method](audio-rollback-crossover-v220-method.md)
