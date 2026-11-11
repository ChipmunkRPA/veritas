# Speech coverage under stronger background noise

The fixed one-second speech fallback raised admissions from **187 to 190 of 195** prepared variants. All three gains occurred with traffic at **−10 dB SNR**. The current gate admitted 7 of 11 genuine clips and 13 of 16 clips from the first synthetic voice family in that condition. With the fallback, those counts rose to 8 and 15. Both rules admitted all 12 clips from the second synthetic voice family. Five variants remained below both gates across the full run. [Frozen method](audio-noise-speech-gate-strong-noise-v269-method.md) · [Exact aggregate](evaluation-panels/audio-noise-speech-gate-strong-noise-complete-v269.json)

| Condition | Prepared | Current gate | With fallback |
| --- | ---: | ---: | ---: |
| Gain-matched clean | 39 | 39 | 39 |
| Kitchen at −5 dB | 39 | 39 | 39 |
| Kitchen at −10 dB | 39 | 38 | 38 |
| Traffic at −5 dB | 39 | 39 | 39 |
| Traffic at −10 dB | 39 | 32 | 35 |

The 195 variants reused 39 utterances and the same two background recordings as the earlier transfer. Another 45 planned variants had no available foreground. The 102 background-only cuts belong to the earlier run, where both rules rejected every cut. Those cuts lack per-cut listening labels. The 18,330 model calls in this run measured speech activity only. A follow-up scored authenticity for the three rescued clips and their comparison group. [Earlier transfer result](audio-noise-speech-gate-transfer-v267-result.md) · [Downstream comparison](audio-noise-speech-rescue-downstream-v270-result.md)
