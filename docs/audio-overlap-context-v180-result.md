# Competing-speech context coincides with nine genuine AI false flags

The isolated CPU worker completed all 98 frozen genuine PACC-P inputs
in 4.75 seconds, using 145.2 MiB maximum sampled process-tree RSS.
Each input contributed one segmentation forward and 175 source-supported
frame cells. The run reused the original speech decisions and had no
guard or sampling errors.
[Frozen method](audio-overlap-context-v180-method.md) · [Verified aggregate](evaluation-panels/audio-overlap-context-result-v180.json)

| VCTK condition | Existing false AI flags | Inputs with competing-speech marker | False AI flags with that marker |
| --- | ---: | ---: | ---: |
| Clean | 0/10 | 0/10 | 0 |
| Babble, 0 dB | 7/10 | 8/10 | 5 |
| Babble, +5 dB | 4/10 | 8/10 | 4 |
| Environmental noise, 0 dB | 0/10 | 0/10 | 0 |
| Environmental noise, +5 dB | 0/10 | 0/10 | 0 |
| Instrumental music, 0 dB | 1/10 | 2/10 | 0 |
| Instrumental music, +5 dB | 0/10 | 1/10 | 0 |

Across the four AMI meetings, the marker appeared on four 0 dB babble
variants, three +5 dB babble variants and one variant in each other
condition, including clean. The recorded treatments establish the mix
conditions. Frame-level true overlap is unannotated. A marker in a music
or clean condition can reflect actual speech overlap or a model error.

The marker coincided with nine of the twelve existing false AI flags.
Using it as an abstention rule would also remove an existing original
conclusion on one +5 dB music input if applied in both directions.
Both observations use the already-consumed genuine control panel.
Before using overlap evidence to change a speech verdict, compare its
effect on matched genuine and synthetic foregrounds under the same
competing speech. The marker supplies timeline context and no origin
judgment.
[Prior voice decisions and source dependence](audio-pacc-false-alarm-v177-result.md)
