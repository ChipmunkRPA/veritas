# Matched Turkish source and recorded background panel

Freeze 24 ordered triplets with three foreground arms. Use all 24 planned v246 RUEG speaker groups, including any unavailable slots. Use the first 24 v236 groups in their frozen roster order. Use the 24 qualified Okur utterances in v240 and then v245 order, with one stock voice group across both stages. The source-preparation receipts must reproduce the published aggregates. Check selected WAV and exact first-three-second PCM fingerprints against the consumed v217 journal and against every other selected foreground. Keep source group and digest bindings private. The known group counts describe the source records. RUEG participant ages are unverified. [Alania preparation](audio-alania-followup-v236-result.md) · [Okur pilot](audio-okur-turkish-v240-result.md) · [Earlier matched method](audio-fresh-shared-noise-v217-method.md)

The five conditions are level-matched clean, kitchen at 0 and +5 dB SNR, and traffic at 0 and +5 dB. Each triplet uses the exact same two three-second background excerpts across its available arms. Only excerpts listed in the private v247 machine-screened provisional packet are eligible. The packet must match the published v247 commitment and the exact v234 qualification, decoded sample offset and signed-16-bit PCM hash. The screen does not supply a human listening review. Interleave recording groups in ordinal order, with each recording's cuts in source order. Cycle the list only when fewer than 24 cuts are available for a band. Report distinct recording and excerpt counts and reuse separately. [Verified mechanical background preparation](audio-commons-background-v234-result.md)

Use the v216 common-gain arithmetic and v217 exact PCM prefix and WAV writer. Within a triplet, each noisy condition adds the same quantized residual bytes to its level-matched clean arm. The measured SNR must stay within 0.1 dB of the target. An unavailable foreground keeps all five rows. A mix that fails the frozen arithmetic keeps the applicable rows unavailable. Replay every saved waveform and its private input binding before publishing aggregate coverage. [Arithmetic](audio-shared-noise-v216-arithmetic.md)

The stage prepares waveforms with the existing data-only guard. It performs no detector scoring or fitting. The same stock voice supplies 24 correlated utterances, and one background recording can supply many clips. The panel supports an exploratory comparison under provisional background evidence. Keep the source and tape grouping available for any later evaluation and report known dependence with results.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.audio_matched_background_panel_v248 preflight
backend/.venv/bin/python -B -m scripts.audio_matched_background_panel_v248 freeze
backend/.venv/bin/python -B -m scripts.audio_matched_background_panel_v248 run
backend/.venv/bin/python -B -m scripts.audio_matched_background_panel_v248 readout
```
