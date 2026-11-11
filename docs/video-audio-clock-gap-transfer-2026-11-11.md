# Video timing across audio clock gaps

Chipmunk Verify now gives the exact source times of the first verified audio timestamp discontinuity when it cannot assess audio/video timing. A gap can arise during recording or delivery. The app does not interpret it as a synchronization anomaly or AI evidence. In the local example, the audio source clock jumps forward from 6.200000 to 6.400000 seconds. The detector and evidence channel agree on those times, and the frozen evidence verifier rejects an altered timestamp. Earlier signed evidence with the original general message still verifies.

## Matched test

Eight 16-second camera and headset cuts came from two AMI recording sessions. Six were retained, and two came from additional time intervals in those sessions selected by published speech annotations. Each cut supplied seven matched files. The picture stream was copied unchanged into each file. The normal audio clock was either continuous or omitted all samples in the half-open **8.000–8.640 s** interval. The genuine and known 320 ms delayed source tracks were included with both clocks. Constant delay and an unrelated headset from the other session supplied harder controls. The local displacement occupied 3.000–6.000 seconds in four cuts and 10.000–13.000 seconds in four. One initial cut exceeded the camera's exact recorded length. The remaining source times were ranked by the same annotations within that recording, and only the failed cut was replaced. The first 49 completed views were kept. [AMI data and license](https://groups.inf.ed.ac.uk/ami/download/) · [Publisher camera and headset mapping](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml) · [Consent](https://groups.inf.ed.ac.uk/ami/corpus/ethicsandconsent.shtml)

The worker checked all 3,200 source RGB positions, the 22,400 matched encoded video packet positions, and 14,008,320 decoded audio samples. The experimental path passed each contiguous, forward-moving audio span independently to the same timing check. Each audio lag search and picture window stayed inside its own verified span, and no comparison crossed the missing 640 ms. The app's existing behavior abstains on those complete tracks. A localization required a flagged time range with intersection-over-union of at least 0.20 against the known local delay. [Previous timing and attribution evaluation](video-speaker-attribution-transfer-2026-11-11.md)

| Gapped condition | Views | Current measured views | Separate-span measured views | Separate-span flags | Localized delay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original soundtrack | 8 | 0 | 5 | 1 | — |
| Local 320 ms delay | 8 | 0 | 5 | 1 | 0/8 |
| Constant 320 ms delay | 8 | 0 | 4 | 1 | — |
| Unrelated off-screen voice | 8 | 0 | 5 | 1 | — |
| Total | 32 | 0 | 19 | 4 | 0/8 |

Separate-span scoring produced **121** fully bounded windows and gave nine views at least four consecutive windows. The one flagged local-delay view produced two review ranges and had a combined overlap of **0.148** with the known edit. It introduced three flags among the 24 gapped controls. The current detector also localized **0/8** delays in the continuous reference files and flagged one continuous original. The two fresh intervals supplied eight gapped views. Six became measurable and none flagged. The additional coverage did not improve delay localization, and separate-span scoring stays offline. The test concerns timing changes on genuine video and mechanically displaced audio. It cannot estimate detection of a well-synchronized generated mouth. [Aggregate counts and commitments](evaluation-panels/video-audio-clock-gap-transfer-2026-11-11.json)
