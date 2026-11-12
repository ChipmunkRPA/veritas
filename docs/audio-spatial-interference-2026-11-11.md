# Stereo interference cancellation for voice screening

Tested on November 11, 2026. Both decision rules failed their frozen advancement criteria. Chipmunk Verify continues to score the decoded mono track with the incumbent voice calibration.

## Method

The check uses half of the stereo channel difference as a background reference. It estimates a complex subtraction coefficient at each frequency using the other two seconds of the same three-second clip. It then applies the estimate to the held second. This preserves the phase of the reference and assumes the target voice is centered with matching waveforms in both channels. The check uses the existing voice classifier, speech activity gate and calibration. The alternative can only downgrade an AI result to inconclusive or raise an inconclusive result to leaning AI, under two separately evaluated policies. The original-file and unassessed states are left alone.

The 217 cases comprise 101 archived exposures and 116 new stereo exposures. The latter use 16 foreground groups absent from the archived stereo panel, three synthetic voice families, and two AMI meeting rooms with separately recorded microphones. The foreground groups and rooms were used in the prior mono study. The new stereo transfer has 64 centered mixes, 32 displaced-voice mixes, 16 clean controls and four native room controls. The displaced mixes delay one channel by four samples and reduce it by 3 dB. Each room uses sequential and simultaneous speech intervals as interference. The clips are controlled mixes with known components. Performance on arbitrary spatial arrangements and unknown generator families remains unmeasured.

## Results

| Transfer condition | Rule | New true AI flags | New genuine false flags | Lost true AI flags | Cleared genuine false flags |
| --- | --- | ---: | ---: | ---: | ---: |
| Centered, 36 synthetic and 28 genuine | Recover inconclusive cases | 3 | 4 | 0 | 0 |
| Centered, same cases | Withhold weak AI flags | 0 | 0 | 3 | 0 |
| Displaced, 18 synthetic and 14 genuine | Recover inconclusive cases | 2 | 1 | 0 | 0 |
| Displaced, same cases | Withhold weak AI flags | 0 | 0 | 1 | 0 |

The centered recoveries span three foreground groups, two generator families and both rooms. The centered false alarms rose from 6/28 to 10/28 under that rule. Neither rule changed a clean or native control decision. Across the 32 matched scene pairs, median SNR gain was +1.41 dB for centered voices and −0.52 dB for displaced voices. The median within-pair change was −1.26 dB. Displacement also reduced the median voice level by 0.82 dB because some of the voice entered the channel-difference reference.

The offline run made 291 classifier forwards and processed 26,508 speech-activity frames. The [aggregate receipt](evaluation-panels/audio-spatial-interference-2026-11-11.json) records the exact freeze and score hashes, group counts, policies, and acoustic metrics. Inputs and per-case responses stay local. Use the `prepare`, `score`, and `aggregate` stages in [the runner](../backend/scripts/audio_spatial_interference.py). Reproduction uses the bound archived paired-stereo panel, the earlier mono dereverberation panel, and AMI Array1 channels 01 and 05 for ES2007c and IS1004c.

An [independent replay](../backend/scripts/review_audio_spatial_interference.py) rebuilt all 116 transfer scenes and verified all 217 case bindings, 166 conditioned waveform hashes, 164 acoustic measurements, and the decision tallies from the saved responses. It ran without inference. One archived waveform needed the frozen SciPy synthesis convention to resolve a half-integer rounding tie.

Method references include [geometry-informed enhancement with cross-channel interference cancellation](https://arxiv.org/abs/2503.03593) and [parametric multichannel Wiener filtering](https://arxiv.org/abs/2507.13863). The runner implements its own small filter and imports no third-party filter code or new weights. AMI recordings are available under the [AMI Corpus license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml).
