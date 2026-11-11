# Stereo decision transfer on natural meeting audio

The frozen AI-only stereo rule and mono gave identical results across 24 genuine recording windows from two AMI meetings. The rule selected a physical channel in one window, where mono already passed the speech gate. Across both meetings it added zero assessed windows and zero false AI flags. [Frozen method](audio-ami-stereo-transfer-v293-method.md) · [Exact aggregate](evaluation-panels/audio-ami-stereo-transfer-complete-v293.json)

| Outcome | Mono | AI-only stereo rule |
| --- | ---: | ---: |
| Leaning original | 5 | 5 |
| Leaning AI on genuine input | 0 | 0 |
| Inconclusive | 4 | 4 |
| Unassessed by the speech gate | 15 | 15 |
| Total | 24 | 24 |

The two meetings contributed 2 and 7 assessed windows out of 12 each. All 24 windows remained in the comparison, including quiet sections. The classifier scored nine mono views and one selected-channel view. Both stages finished locally with the existing resource limits and the saved readout replayed identically.

The zero false flags apply to nine assessed windows across two meeting sessions. The other 15 windows remain unassessed. This genuine-only pilot adds evidence about coverage on natural recordings and leaves synthetic recall unmeasured. A natural comparison with more channel imbalance and matched synthetic speech is still needed to measure when the rule helps. The result does not change serving or overall confidence. [Source preparation](audio-ami-native-stereo-v291-result.md) · [Controlled stereo holdout](audio-native-stereo-holdout-v284-result.md)
