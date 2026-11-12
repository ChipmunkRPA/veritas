# Native speech timing for the untouched GRID reserve

All 16 reserve originals from eight source groups now have verified publisher word intervals on their native video and embedded audio clocks. The audit recovered eight references that were unavailable under literal directory matching. The new source bank qualifies timing and ordinary editing controls for a future lip-sync study. It contains zero new neural redraws or independently confirmed phoneme intervals. [Public qualification and replay](evaluation-panels/grid-reserve-word-clocks-2026-11-12.json)

## Source and timing

The official GRID release supplies endpointed 25 kHz WAVs and word alignments under CC BY 4.0. Its paper also describes separate phone-level forced alignment. The released word archive used here supplies no independently annotated phone boundaries or releases for these clips. [Publisher record](https://zenodo.org/records/3625687) · [Corpus and paper](https://spandh.dcs.shef.ac.uk/gridcorpus/)

For each retained video archive, compare its complete set of 1,000 distinct utterance stems against every official word directory. Each video archive had exactly one full match and zero overlap with every other directory. Four groups match their nominal label. Four match a differently numbered word directory, consistent with the correction in the Lhotse recipe. The study derives the match from the publisher inventories and independently checks the speech waveforms. [Documented directory correction](https://lhotse.readthedocs.io/en/latest/_modules/lhotse/recipes/grid.html#download_grid)

A bounded retrieval transferred 10,134,708 bytes from the official ZIP, including its directory and the 16 selected WAV entries. Each extracted entry passed its ZIP CRC. Every endpointed WAV has the six-word span plus 250 samples of leading and 250 samples of trailing context at 25 kHz, and one inclusive endpoint sample. The complete waveform and each word were independently correlated with the re-decoded embedded soundtrack after resampling to 16 kHz. Full-waveform correlations ranged from **0.929 to 0.992**. All 96 word correlations reached **0.900 or higher**. The measured clock offset was **15–17 samples**, or **0.94–1.06 ms**, and local word matches differed from the shared offset by at most **3 samples**.

The verifier reopened **1,200 native pictures** and **762,368 original canonical PCM samples**, checked the retained source hashes and timing, and projected **96 publisher word intervals** into the exact rational source clock. Each result records the first included audio sample and the first native picture at or after each boundary. The waveform residual measures clock correspondence. The original word annotations retain their own boundary uncertainty.

The earlier execution wrapper correctly refuses code drift in the historical audio/video decoder bindings. This study verifies the completed plan and its retained artifact hash, then independently reopens every original with the current decoders. The archived run and its freeze stay intact. [Historical plan](grid-transfer-plan-v143-method.md) · [Prior native verification](grid-native-input-review-v142-result.md)

## Fresh ordinary mouth controls

The first directly matched source and first remapped source each supplied one fixed word window. The two matched treatments replace the mouth area with an eye-registered patch from the same recording **320 ms earlier**, or apply mild brightness to that same region. They use original source pixels and a fixed feathered region. These authored edits are ordinary controls.

| Qualified source | Local comparison interval | Before, during and after frames | Treatment results |
| --- | --- | --- | --- |
| Direct directory match | 1.160–1.400 s | 28, 29, 35 | Exact interval in both controls |
| Remapped directory match | 1.400–1.600 s | 34, 35, 40 | Exact interval in both controls |

The four lossless deliveries preserved all **300 native picture timestamps** and **190,592 soundtrack samples**. Replay rebuilt **22 exact changed-picture masks**. The existing earlier-copy comparison recovered all four ranges and supplied 24 before, during and after pictures across the earlier and edited files. It found zero audio differences. The result establishes exact ordinary editing and evidence correspondence. It supplies no AI detection result or customer confidence. [Existing earlier-copy comparison](video-reference-comparison-2026-11-11.md)

New neural positives require a reserve-scoped render run using the existing rights-reviewed MuseTalk or LivePortrait assets. The current renderer freezes admit pilot rows. The next evaluation also needs independently checked phone-level or visible closure labels to measure achieved synchrony. The verified word windows, retained complete original clocks and tested ordinary geometry now supply the source intake and matched controls. [Previously qualified renderers](video-timing-coverage-and-codec-holds-2026-11-11.md) · [Native renderer and construction](grid-native-clips-v149-method.md)

## Reproduce

Run with the retained source bank and the official word-alignment archive. The WAV fetcher reads only the archive directory and the selected entries. All media and per-source records remain local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.fetch_grid_reserve_word_wavs "$wav_selection"
backend/.venv/bin/python -B -m scripts.grid_reserve_word_clocks "$wav_selection" "$qualification"
backend/.venv/bin/python -B -m scripts.build_grid_word_timed_controls "$qualification" "$controls"
```
