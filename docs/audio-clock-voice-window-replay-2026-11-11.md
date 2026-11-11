# Voice checks across source timestamp breaks

The decoder writes a continuous listening WAV and separately records the source timestamp of each decoded span. Before this change, the voice adapter could interpret the concatenated samples as one continuous recording. A missing or repeated section of the source timeline could receive the same whole-file voice direction and confidence as a continuous clip. This concerns what the voice result covers. A timestamp gap or overlap has no direction or weight as evidence of synthesis. [Exact aggregate](evaluation-panels/audio-clock-voice-window-replay-2026-11-11.json)

## Frozen paired replay

Use 14 retained, known-origin three-second clips. Four English and four RUEG recordings supply genuine speech. Alania, Drifting and Okur each supply two synthetic clips. Every exact utterance is outside the earlier stereo confirmation roster. The source banks had older project exposure, and some stock synthetic voices recur. A generated tone was used to verify remux and clock arithmetic before the freeze. The 42 media variants represent 14 distinct utterances. [English source rights](audio-stereo-component-transfer-2026-11-11.md) · [Other foreground source rights](audio-spectral-texture-transfer-2026-11-11.md) · [Earlier stereo confirmation](audio-frequency-separation-transfer-2026-11-11.md)

Remux each clip to lossless PCM Matroska in three conditions. Keep the original continuous timestamps, introduce a 640 ms gap, or introduce a 64 ms overlap. The break is always at decoded sample 24,576, or 1.536 seconds. Each variant must decode to precisely the same 48,000 samples at 16 kHz. Run the actual local candidate adapter in Quick with the same speech admission, voice model and calibration. Independently decode the source clock, count crossing windows, and compare exact original waveforms, calibrated directions and retained raw window responses.

## Local results

| Original clock | Whole-file estimates before → after | Crossing voice windows before → after | Recorded voice windows after |
| --- | ---: | ---: | ---: |
| Continuous | 13 → 13 | 0 → 0 | 13 |
| 640 ms gap | 13 → 0 | 13 → 0 | 13 |
| 64 ms overlap | 13 → 0 | 13 → 0 | 13 |

All 13 assessed continuous clips preserved their exact raw score and calibrated direction. Seven leaned original and six leaned AI. The eighth genuine clip had insufficient speech for the check in every variant. Before the guard, both broken-clock variants reproduced all 13 whole-file directions and crossed the break. With the guard, 13 retained individual-window raw results for each break condition. All 26 broken files with raw results returned an inconclusive whole-file voice assessment without a calibrated estimate. The two broken variants of the unassessed source remained unassessed.

The original arm made 86 real classifier forwards, including stability probes and the within-interval comparison. It completed in 56.70 seconds with 2.99 GB peak process RSS. The guarded replay made 46 real classifier forwards in 29.58 seconds with 3.09 GB peak RSS. Every guarded response matched the frozen within-interval comparison. A separate readout reproduced the combined aggregate and checked each file's decoded input, source timestamp jump, stored model response and model-call total.

## Behavior in the app

Audio pipeline a32 and video pipeline p47 split voice windows and speech activity at a source timestamp gap or overlap of at least 20 ms. Existing profile caps still apply across all eligible windows. The app can show retained interval-level observations on either side and withholds a recording-wide voice estimate. Other video checks can still support a visual assessment. The PDF and signed JSON include the exact source spans and indicate when the current guard withheld voice confidence. Historical results keep the crossings they originally recorded. Prior signed r83 and r84 reports remain available.

An isolated Full-profile app test uploaded two supported ALAC M4A files with the same 48,000 decoded samples. Only one had a 640 ms source gap at sample 24,576. The continuous upload retained its calibrated AI direction. The interrupted upload scored the two intervals on either side, recorded zero crossing windows, and showed an inconclusive recording-wide assessment with no confidence percentage. The JSON export matched the evidence endpoint. Both PDFs rendered, and both signed packages verified against the original uploads, including the exact voice clip samples. The check used simulated $9.99 Stripe payments and verified that the worker started analysis only after payment verification.
