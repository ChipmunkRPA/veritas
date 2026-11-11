# Speech activity in a second measured room

The earlier two-room preflight measured no qualifying channel reversal at Surrey’s ±90° side positions on six source groups. The BBC room qualified, but its speech-activity check recovered none of the six mono rejections. This follow-up measured another set of documented Surrey positions and tested whether stronger channel separation helps the existing speech gate. [Earlier room comparison](writing-and-room-transfer-2026-11-11.md)

## Measured geometry and source pairing

Jon Francombe’s University of Surrey guide identifies MATLAB channels 5 and 6 as back left at +135° and back right at −135°, both at 0° elevation. Channel 3 is front center. The guide defines the left and right receiver columns and the one-based loudspeaker order. The study used the exact previously qualified 48 kHz response file at 0° head orientation, without trimming, shifting, or normalizing the ears separately. The guide’s SHA-256 was checked against its earlier recorded digest. The measured data are CC BY 4.0. [Publisher dataset and attribution](https://doi.org/10.15126/surreydata.00813511) · [Pinned guide, Table 2 and §3.2](https://github.com/IoSR-Surrey/IoSR_ListeningRoom_BRIRs/blob/d36af1bbd0f4083a9f9ee678ff8f499044e851a7/IoSR_ListeningRoom_BRIR.pdf)

An initial rear-pair preflight prepared 30 cases. All 18 prior clean and side cases were reproduced byte for byte, and the 12 rear-noise cases kept the same clean speech samples. Rear noise reversed the cleaner channel by at least 3 dB in zero of six groups. The impulse peaks arrived earlier on the expected side in each direction. Those early differences did not predict enough broadband separation in the actual mixtures.

A separately frozen four-pair comparison kept speech at front center and used matched clean, left-noise, and right-noise cases at a joint −10 dB signal-to-noise ratio. Each role contained two genuine, two expressive synthetic, and two stock synthetic foreground groups, paired with two role-local MUSAN noise recordings. The two roles share no foreground group, declared voice, source or decoded foreground, or background source. The foregrounds and noise had been used previously. The holdout role tests transfer across source groups within this study, and it matches the previously measured BBC inputs. Source rights are recorded in the earlier qualification. [Source rights and provenance](audio-short-edit-source-plan-v169.md)

| Noise pair | Development reversal /6 | Holdout reversal /6 |
| --- | ---: | ---: |
| Front inner ±30° | 0 | 3 |
| Front wide ±60° | 3 | 3 |
| Side ±90° | 0 | 3 |
| Rear ±135° | 0 | 2 |

The fixed rule required at least three groups with a reversal of at least 3 dB in each direction. Only ±60° met that rule on development. The held-out source and noise groups then qualified the same choice. At ±60°, joint SNR was −10.0002 to −9.9981 dB across the two roles. Every matched case used a three-second zero-origin clock and passed stereo WAV, individual channel, and production mono decoder verification. [Exact geometry and activity aggregate](evaluation-panels/audio-measured-surrey-activity-transfer-2026-11-11.json)

## Speech-activity results

The study ran the existing v279 active-to-quiet contrast selector and production full-profile speech window check. Each case supplied the production mono input and the two decoded native channels. The selector had no access to the known source origins, noise levels, or cleaner-channel labels. All 36 cases and 10,152 speech-activity forwards completed in the verified CPU retry. The first attempt wrote all observations and stopped on an inherited summary path. The corrected retry produced a byte-identical observation journal. A separate readout repeated the PCM, source-clock, channel-swap, and aggregate checks. All 19 focused existing tests passed. [Existing gate method](audio-native-stereo-gate-v279-method.md)

| View group | Mono admission | Mono or fixed selected channel |
| --- | ---: | ---: |
| Surrey development | 3/18 | 3/18 |
| Surrey held-out sources | 8/18 | 8/18 |
| Surrey clean controls | 7/12 | 7/12 |
| Surrey strong-noise cases | 4/24 | 4/24 |
| BBC on the same held-out sources and noise | 12/18 | 12/18 |

The selector abstained on all 36 Surrey cases. Of 25 mono rejections, both native channels rejected 23. The other two were left- and right-noise versions of one genuine holdout source. The physically clearer ear admitted each case, with SNR gaps of 9.16 and 10.15 dB. The selector’s measured contrast gaps were 1.69 and 0.14 dB, below its fixed 3 dB criterion. Those channel-only admissions were diagnostics. The app’s selector recovered neither case. On the same 18 holdout inputs, four cases admitted in BBC were rejected in Surrey, eight were admitted in both, and six were rejected in both. [BBC activity aggregate](evaluation-panels/audio-measured-bbc-activity-transfer.json)

These measurements describe window admission under simulated mixtures using measured room responses. They provide no accuracy or confidence estimate for synthetic speech. The study ran zero authenticity forwards. The serving rule remains unchanged.
