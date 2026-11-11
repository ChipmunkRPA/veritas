# Language diagnostic identifies all 38 frozen short speech clips

The guarded CPU worker completed 38 first-language-token forwards in
9.20 seconds, with 807.5 MiB maximum sampled process-tree RSS and no
guard or sampling errors. It generated no transcripts.
[Frozen method](audio-language-scope-v184-method.md) · [Verified aggregate](evaluation-panels/audio-language-scope-result-v184.json)

| Frozen source | Clips | Source groups | Identified English | Identified Turkish | Other or unavailable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Genuine English speech | 14 | 7 | 14 | 0 | 0 |
| Synthetic Turkish speech | 24 | 24 | 0 | 24 | 0 |

Both Turkish clips that the English-calibrated speech check had called
original were identified as Turkish in this diagnostic. Each top language
score exceeded the runner-up by at least 0.5 on the normalized model
response scale. This margin is descriptive and supplies no customer
confidence or decision rule.

Language, source corpus and speech origin differ together across the
two groups. A separate comparison with genuine Turkish and synthetic
English is needed to check language coverage across both origin classes.
The diagnostic provides evidence about which language was spoken on
these inputs. Serving policy has not been changed.
[Turkish transfer errors](audio-alania-transfer-v179-result.md)
