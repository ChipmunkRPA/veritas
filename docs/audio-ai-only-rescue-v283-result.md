# AI-only stereo rescue on development pairs

On 20 synthetic speech cases, the fixed rescue rule raised correct AI directions from **4 to 7**, across three distinct added source groups. Two additions came from inconclusive mono decisions and one came from an unassessed mono input. All three existing false original directions remained. The rule left all 13 genuine cases in their mono dispositions, with zero false AI flags. [Frozen method](audio-ai-only-rescue-v283-method.md) · [Exact aggregate](evaluation-panels/audio-ai-only-rescue-complete-v283.json)

| Policy | Genuine correct / false AI (13) | Synthetic correct / false original (20) |
| --- | ---: | ---: |
| Mono | 7 / 0 | 4 / 3 |
| Selected channel alone | 10 / 0 | 7 / 4 |
| Mono with AI-only rescue | 7 / 0 | 7 / 3 |

The replay used the exact 33 saved development pairs and made no additional model calls. The selected-only policy recovered more genuine original directions and produced one more false original direction on synthetic speech. The AI-only rule trades those original-direction gains for a more conservative result. The 200-case holdout provides the next test of transfer beyond these deliberately imbalanced cases. [Saved classifier comparison](audio-native-stereo-authenticity-v282-result.md)
