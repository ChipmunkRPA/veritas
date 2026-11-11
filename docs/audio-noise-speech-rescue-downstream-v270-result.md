# Audio authenticity after speech admission in strong traffic noise

All three clips recovered by the speech fallback were **inconclusive** at −10 dB traffic noise. The classifier correctly leaned original for the genuine clip and leaned AI for both synthetic clips when it scored their matched clean versions. The fallback expanded coverage for these clips without recovering a useful authenticity direction. [Frozen method](audio-noise-speech-rescue-downstream-v270-method.md) · [Exact aggregate](evaluation-panels/audio-noise-speech-rescue-downstream-complete-v270.json)

| Speech group | Source groups | Clean direction | Direction at −10 dB |
| --- | ---: | --- | --- |
| Genuine, fallback rescue | 1 | 1 original | 1 inconclusive |
| Genuine, already admitted | 7 | 7 original | 7 inconclusive |
| Synthetic, fallback rescue | 2 | 2 AI | 2 inconclusive |
| Synthetic, already admitted | 13 | 12 AI, 1 inconclusive | 12 inconclusive, 1 original in error |

The local classifier scored all 46 waveforms with 56 forwards, including the existing stability probe. Across the 23 noisy waveforms, 22 were inconclusive and one synthetic waveform leaned original. No genuine waveform leaned AI. This comparison conditions on speech admission, uses one traffic recording and covers one synthetic generator family. Keep the fallback experimental. The run leaves production and its confidence calculation unchanged. [Speech admission results](audio-noise-speech-gate-strong-noise-v269-result.md)
