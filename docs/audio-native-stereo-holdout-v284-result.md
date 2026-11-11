# Held-out accuracy of the fixed stereo rescue

The fixed AI-only rule detected **49 of 120 synthetic variants**, compared with **43 of 120** for mono. It added one false AI flag, taking the genuine false flags from **1 to 2 of 80**. All seven changes came from deliberately imbalanced mixes. Clean and equal-gain outcomes were identical across the rules. The six correct additions came from six connected source groups, with one clone, one expressive, and four stock additions. [Frozen method](audio-native-stereo-holdout-v284-method.md) · [Exact aggregate](evaluation-panels/audio-native-stereo-holdout-complete-v284.json)

| Outcome across 200 cases | Mono | AI-only stereo rescue | Channel-priority diagnostic |
| --- | ---: | ---: | ---: |
| Correct synthetic direction | 43 | 49 | 49 |
| Correct genuine direction | 47 | 47 | 51 |
| Genuine falsely leaning AI | 1 | 2 | 2 |
| Synthetic falsely leaning original | 8 | 8 | 8 |
| Inconclusive | 74 | 67 | 66 |
| Unassessed | 27 | 27 | 24 |

The speech gate admitted 173 cases from mono alone and 176 with a selected channel. All six added correct AI directions came from cases where mono was already assessed and inconclusive. The complete classifier stage scored 218 admitted views in 291 forwards and reported all 200 cases. The channel-priority diagnostic recovered four more genuine directions than the AI-only rule. Its unchanged total false-original count includes an increase on the expressive family and a decrease on the stock family.

The holdout contains 40 foreground recordings across 32 connected groups and two background sessions. Its equal-gain and imbalanced mixtures combine one centered speech track with recorded stereo background. The measured gain supports testing the rule on naturally recorded stereo with channel imbalance. Production confidence and the serving route are unchanged. [Development replay](audio-ai-only-rescue-v283-result.md) · [Stereo input construction](audio-native-stereo-pairs-v277-result.md)
