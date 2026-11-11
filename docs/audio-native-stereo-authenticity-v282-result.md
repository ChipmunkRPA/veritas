# Authenticity decisions after stereo speech recovery

On the seven clips that the current mono gate missed, the selected channel produced **three correct directions**, one wrong direction and three inconclusive results. Two genuine clips leaned original and one synthetic clip leaned AI. One other synthetic clip incorrectly leaned original. The mono gate left all seven unassessed. When scored for the diagnostic counterfactual, all seven mono mixes were inconclusive. [Frozen method](audio-native-stereo-authenticity-v282-method.md) · [Exact aggregate](evaluation-panels/audio-native-stereo-authenticity-complete-v282.json)

| Already admitted comparison | Count | Correct on mono | Correct on selected channel | Wrong on mono | Wrong on selected channel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Genuine | 10 | 7 | 8 | 0 | 0 |
| Expressive synthetic | 16 | 4 | 6 | 3 | 3 |

All 84 planned waveforms completed in 115 classifier forwards. The gains come from deliberately asymmetric recordings in two development backgrounds, one synthetic family and reused source groups. The first useful directions on the rescued clips support a closer test of when channel evidence should supplement mono. The false original result is a concrete reason to compare an AI-only rescue rule with full channel replacement before changing serving. The 200 holdout cases remain unscored. [Stereo speech coverage](audio-native-stereo-gate-v279-result.md)
