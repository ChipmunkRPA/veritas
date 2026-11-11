# Brightness context around retained video flags

The readout verified **471 native frame observations across 12 original appearance flags**. In six genuine webcam renders, the fraction of sampled nonface sites with a sustained temporary luma shift of at least 12 code values ranged from **0.24% to 16.08%**. That fraction was zero in four renders of one known swap and in two other no-swap renders. The latter two still had median temporary context shifts of about 8 to 9 code values. [Frozen method](video-luma-context-v278-method.md) · [Exact aggregate](evaluation-panels/video-luma-context-complete-v278.json)

| Cohort | Flags measured | Nonface sites reaching 12 code values |
| --- | ---: | ---: |
| Genuine webcam, two source groups | 6/6 | 0.24%–16.08% |
| Known swap, one intervention | 4/4 | 0% |
| No-swap renders matched to that intervention | 2/2 | 0% |

None of the measured face shifts reached 12 units, so the same-direction fraction was unavailable. Brightness context differed between the webcam and swap renders on this diagnostic, with two other false flags still present at zero measured support above 12. Renders share sources and the grid has no camera-motion registration. This result leaves the current flagging policy in place. [Earlier selected-window audit](video-saved-landmark-support-v254-result.md)
