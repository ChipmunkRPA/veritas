# Voice corroboration on two additional strong backgrounds

The previous bandpass check removed two false AI flags when replayed on the original failures. This test keeps both corroboration rules fixed and uses two different, predominantly in-band backgrounds with more genuine speech admitted by the normal gate. [Earlier comparison](audio-background-corroboration-2026-11-11.md) · [Exact transfer results](evaluation-panels/audio-frozen-corroboration-transfer-2026-11-11.json)

## Fixed local comparison

Use 18 genuine recordings from 13 declared reader or participant groups and 12 synthetic recordings across cloned, expressive, and stock speech. Each recording has one gain-matched clean view and two −5 dB background mixtures. The foreground and background retain their full three-second PCM length, and all added residuals reproduce the recorded noise up to signed-sample rounding. Each isolated background is tested at its original gain and one predetermined mix gain. This yields 90 matched speech views and four background-only controls.

The noise recordings come from the declared public-domain MUSAN subset and were outside the earlier corroboration and room panels. They have 96.8% and 75.5% of their measured power in the fixed 100–3,800 Hz analysis band, compared with 67.6% for the background that drove the earlier failure. The foregrounds exclude known prior corroboration and room source links. LibriSpeech, RUEG and the synthetic banks retain their reviewed evaluation permissions. Both foreground banks had earlier baseline exposure. This is a new frozen-policy comparison with distinct inputs relative to the earlier corroboration, with unknown model pretraining overlap. [Short-speech source and rights](audio-short-edit-source-plan-v169.md) · [RUEG source and rights](audio-rueg-genuine-v246-method.md)

Keep the original voice classifier, calibration and production speech window. Test the same bandpass and quiet-frame checks on measured AI calls. The guard changes an AI call to inconclusive only when the transformed response falls out of the AI decision region and raw log odds drop beyond the existing stability tolerance. The quiet-frame arm preserves the original result if no eligible background estimate is available. The transfer predeclared a larger-confirmation criterion of removing false AI on at least two genuine groups while losing at most one synthetic detection and assessing at least eight genuine speech views per noisy condition. [Frozen rules and earlier results](audio-background-corroboration-2026-11-11.md)

## Results

| Condition | Current genuine false flags / assessed | Current synthetic flags / assessed | Synthetic flags after band / quiet | Unassessed genuine / synthetic |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0/18 | 11/12 | 11 / 11 | 0 / 0 |
| Background C, −5 dB | 4/11 | 8/9 | 8 / 7 | 7 / 3 |
| Background D, −5 dB | 1/16 | 10/12 | 9 / 10 | 2 / 0 |

Both methods preserved all five genuine false flags and each changed one different correct expressive-speech detection to inconclusive. The five false flags came from four LibriSpeech readers. RUEG supplied only two of the 11 assessed genuine views in background C and six of the 16 in background D. None of the four isolated controls passed speech admission. The quiet-frame view was unavailable for two of the false AI views and left the other three AI flags in place.

Neither rule met the advance criterion. The earlier same-input bandpass gain did not carry over to these two backgrounds. The current speech route and confidence remain unchanged. The comparison establishes a repeatable failure on the tested mixtures, with too few background recordings to estimate a wider error rate.

All 94 views reached a recorded outcome. The current speech gate made 8,836 frame forwards. The classifier made 165 forwards across the original and corroboration views in 99.27 seconds, with 2.92 GB peak process memory. The separate audit checked 94 PCM hashes, 60 quantized residuals, all four isolated controls, and every direction and recorded model-call count. Only aggregate results and whole-artifact commitments are published.
