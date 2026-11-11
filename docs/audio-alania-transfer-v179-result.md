# English-reference speech check flags 21 of 24 Turkish synthetic clips

All 24 frozen Alania inputs passed full production decoding, including
the native duration, continuous zero-origin sample map and saved PCM
checks. All passed the speech gate. The guarded CPU scorer completed
in 20.59 seconds with 2.91 GiB maximum sampled process-tree RSS, 24 raw
forwards and two original-side stability forwards. Both stages
completed without guard or sampling errors.
[Frozen method](audio-alania-transfer-v179-method.md) · [Preparation](evaluation-panels/audio-alania-preparation-result-v179.json) · [Verified aggregate](evaluation-panels/audio-alania-transfer-result-v179.json)

| Publisher render mode | Known AI clips | AI tendency | Original tendency | Inconclusive |
| --- | ---: | ---: | ---: | ---: |
| Reference | 18 | 15 | 2 | 1 |
| Reference plus instruction | 6 | 6 | 0 | 0 |
| Total | 24 | 21 | 2 | 1 |

The two original tendencies are directional errors. Both passed the
existing stability check. The third missed AI clip was inconclusive.
These results describe one clean, publisher-filtered synthetic generator
with 24 distinct designed voice and text groups. The frozen route uses
an English reference. This positive-only transfer check estimates no
Turkish false-alarm rate or calibrated confidence.
[Publisher source and frozen roster](audio-alania-roster-v178-method.md)
