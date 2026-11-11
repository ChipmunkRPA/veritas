# Three-second speech features verified on native samples

The CPU worker completed all four encoder calls in 7.16 seconds, with
1.17 GiB maximum sampled worker-tree memory. It used one genuine clip
and one clip from each synthetic generator under 0 dB traffic noise.
Every input used exactly 48,000 original samples and produced 149 native
frames and 4,608 pooled features. The repeat reproduced the selected
hidden-state commitments and pooled values exactly.
[Method](audio-wavlm-three-second-v226-method.md) · [Verified receipt](evaluation-panels/audio-wavlm-three-second-complete-v226.json)

This result qualifies the input and feature clock for a separate test of
the fixed paired head. The four-second English fitting and calibration
remain the comparison reference. [Paired-head results](audio-paired-consistency-v223-result.md)
