# Multiband speech and mouth-shape timing transfer

The new alignment method often proposed a range near an inserted 320 ms audio delay on a controlled test, but its fixed review point recovered none of those edits. It also flagged two unrelated voices. In continuous conversation, the current method localized one of eight timed changes and the new method localized none. The experimental method stays outside the app. [Aggregate results](evaluation-panels/video-lip-spectral-timing-transfer-2026-11-11.json)

## Frozen comparison

CREMA-D supplied 80 exact publisher assets, four scripted utterances per actor. Eight actors supplied development recordings. Twelve actors were selected by fixed hash order from outside the two earlier CREMA-D rosters, with a different set of scripts than development. One of the twelve had an utterance below the minimum native support and was excluded before fitting or scoring. The other eleven supplied 77 matched views. Each view retained exactly the same selected pictures across its seven conditions. The first three recordings were joined on a 25 Hz output clock. Each target used the nearest independently verified native picture within 20 ms, and the input audio was aligned to the first observed picture. Input resampling admitted at most one millisecond of original FLV clock drift. These are composed scripted utterances. The AMI transfer separately used eight continuous 16-second cuts from two prior camera and headset sessions. [CREMA-D publisher terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [AMI terms](https://groups.inf.ed.ac.uk/ami/download/) · [AMI clock and camera alignment](video-audio-clock-gap-transfer-2026-11-11.md)

The candidate learned four paired components by regularized canonical correlation on **1,454** source-bound development observations. It compares the shape and low-frequency texture of the mouth crop, their changes over time, and acoustic power and changes in eight speech bands. It searches for a short range with a distinct lag supported by two surrounding groups that agree with each other. No transcript or new pretrained checkpoint was used. The review point was set from development controls before scoring either evaluation panel. A local flag needs time IoU of at least 0.20 to count as a localization. Correct lag additionally requires the signed estimate to be within 80 ms of the known shift. Both methods abstain on the missing-audio controls. Recent work distinguishes timing from actual articulation. This comparison tests only timing. [PVSync](https://arxiv.org/abs/2610.09223) · [Lip-focused consistency](https://arxiv.org/abs/2610.30872) · [Natural speech timing variation](https://doi.org/10.1371/journal.pcbi.1003743)

| Condition and outcome | Current check | Candidate |
| --- | ---: | ---: |
| CREMA-D local delays localized | 0/11 | 0/11 |
| CREMA-D control timing flags | 0/66 | 2/66 |
| Continuous AMI local delays localized | 1/8 | 0/8 |
| Continuous AMI control timing flags | 3/40 | 1/40 |

CREMA-D controls included recorded sound, level-only changes, constant offsets, unrelated off-screen speech, different words from the same speaker inside a bounded range, and missing audio. The candidate's two CREMA-D control flags were both unrelated off-screen voices. Its AMI control flag was a gain-only view. The three current AMI control flags were one each on recorded, gain-only and constant-delay views. The one current AMI localization had an IoU of 0.223.

Before applying the fixed review point, the best candidate range overlapped eight of eleven CREMA-D delays. Four also had the correct signed lag within 80 ms. On continuous AMI, one raw range overlapped the known edit, and none met both criteria. In a retrospective check on CREMA-D, lowering the point enough to admit the highest-scoring correct timing estimate would also admit five of 66 control flags across three actors, including a recorded view and a gain-only view. This describes the tested score ordering. It supplies no new review point.

The first fit process stopped before fitting when a working-tree helper refactor changed one pinned source hash. The continuation verified and imported the exact original application module in a private directory. The sources, inputs, comparison and fit procedure stayed fixed. Both scoring workers and the independent arithmetic audit exited successfully.

## Independently replayed timing example

The authored **+320 ms** displacement spans **2.240–3.800 s**. The candidate also estimates **+320 ms** and selects **2.400–3.840 s**, with interval IoU **0.875**. Its response of 0.543 is below the frozen point of 0.745, so it produces no finding. An independent lossless replay verified all 160 pictures, 102,400 decoded PCM samples, and six native before, during and after PNGs for the two intervals. The estimated range's pictures are at **2.360, 3.080 and 3.840 s**. Both selected audio extracts independently matched their native-clock sample ranges. The image and audio exhibits remain private. [Offline frame verifier](../backend/scripts/verify_native_video_interval.py)
