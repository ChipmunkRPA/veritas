# A second voice representation under strong traffic noise

The earlier −10 dB traffic study found that three speech-admission rescues all remained inconclusive after the current voice check. The same current raw responses also lost ranking separation under noise. This comparison asks whether the existing WavLM-assisted second opinion recovers some of that loss using the same admitted waveforms. [Earlier downstream result](audio-noise-speech-rescue-downstream-v270-result.md) · [Raw-score replay](audio-stereo-reference-suppression-2026-11-11.md#what-calibration-alone-could-recover)

## Fixed comparison

Keep all 48 planned foregrounds from the earlier strong-noise gate panel, with gain-matched clean and nominal −10 dB traffic views. Reconstruct all 240 earlier clean and noisy mixtures and verify each against its signed-16-bit sample hash. Reuse the earlier gate and its frozen one-second fallback. A foreground enters scoring only when both of its matched views were admitted. The two fixed conditions yield 96 planned views. Eighteen views have no prepared foreground, and eight more are unassessed because their pairs did not pass admission. [Strong-noise source and gate](audio-noise-speech-gate-strong-noise-v269-result.md)

The 70 admitted views cover eight genuine groups, 15 designed-voice synthetic groups, and 12 stock-voice clips from three stock voices. The 46 current results in the earlier downstream study match their exact signed waveform commitments. The remaining 24 stock-voice views require new current scores with the same weights and calibration. The second opinion receives exactly the same 48,000 decoded samples as the current scorer for each admitted input. No other inputs receive a score.

Use the pinned 149-frame WavLM extraction at 16 kHz, the previously fitted English four-second score head, and the previously published English diagnostic operating points. The head combines WavLM features with the current classifier response. The rule lets a candidate AI response replace an inconclusive current result. It retains current AI and original directions, and preserves unassessed status. The head and operating points were fixed before this replay. They supply no calibrated probability for these Turkish, three-second mixtures. [Pinned model card](https://huggingface.co/microsoft/wavlm-base-plus/blob/4c66d4806a428f2e922ccfa1a962776e232d487b/README.md) · [Publisher weight terms](https://github.com/microsoft/UniSpeech/blob/main/LICENSE) · [Frozen native input and score method](audio-paired-transfer-v230-method.md)

The pinned weight terms are CC BY-SA 3.0, and the separately published code notice is MIT. The experiment uses the existing local checkpoint and keeps source WAVs, encoder features, scores, and model weights local. The foregrounds were used in earlier tests at milder noise. This is transfer to a new noise level for the second opinion, with repeated sources across conditions and one traffic recording. The candidate result cannot independently estimate population false-alarm rates or validate a new production calibration.

## Local results

The current model completed all 24 new stock-voice views with 24 forwards in 19.09 seconds and 2.93 GB peak process memory. The candidate completed 71 encoder forwards in 8.91 seconds and 1.30 GB peak process memory, including an exact first-input repeat. A separate audit verified both models’ input commitments, all 96 case outcomes, the second-opinion decisions, and the ranking arithmetic. The candidate stage used the frozen head and ran zero new speech-activity calls. [Exact aggregate](evaluation-panels/audio-wavlm-strong-traffic-2026-11-11.json)

| Noisy −10 dB traffic group | Admitted / planned | Current AI / original / inconclusive | Candidate AI / original / inconclusive | Second opinion AI / original / inconclusive |
| --- | ---: | ---: | ---: | ---: |
| Genuine | 8 / 16 | 0 / 0 / 8 | 0 / 5 / 3 | 0 / 0 / 8 |
| Designed-voice synthetic | 15 / 16 | 0 / 1 / 14 | 1 / 2 / 12 | 1 / 1 / 13 |
| Stock-voice synthetic | 12 / 16 | 1 / 0 / 11 | 3 / 2 / 7 | 3 / 0 / 9 |

The second opinion recovered three noisy synthetic files across three source groups and raised the admitted synthetic flag count from 1/27 to 4/27. No genuine clip received an AI flag among the eight assessed. Eight of the 16 planned genuine clips were unassessed, so this result gives no false-alarm measurement for those clips. The candidate alone assigned a wrong-original direction to four noisy synthetic files. The second-opinion rule retained the current model’s one wrong-original result. Its other 22 noisy synthetic misses were inconclusive. In clean audio the second opinion raised synthetic AI flags from 26/27 to 27/27, and all eight assessed genuine clips leaned original in both arms.

On the noisy admitted clips, raw ROC AUC rose from 0.733 to 0.767 for designed voices and stayed at 0.802 for stock voices. The added flags support further evaluation of the second opinion. A single traffic recording, previously used foregrounds, and only eight assessed genuine files limit a decision about serving it by default.

The full historical runner rejected replay after source-verifier implementation changes. The separate replay recovered every original signed-sample commitment directly without editing the historical verifier. A launch retry corrected duplicate thread setup before any model forward. The second freeze retained the exact waveform, weight, head, and point bindings. [Prior source and gating method](audio-noise-speech-gate-strong-noise-v269-method.md)
