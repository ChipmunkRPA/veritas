# Compare an audio recording with an earlier copy

On a local development instance, set `VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED=true`. Sign in, open a completed audio analysis, and choose **Compare with an earlier audio recording**. The service compares the retained analyzed file against a temporary upload. It creates no analysis, payment order, or retained copy of the new upload.

Each file may contain up to 30 seconds of decoded audio and occupy at most 48 MiB. Both first audio streams must yield continuous signed 16-bit, 16 kHz mono samples with matching start ticks and sample counts. An incompatible clock or duration produces an unassessed comparison. The tool measures differences at matching sample positions. It does not estimate alignment, separate speakers, or determine AI origin.

The report groups adjacent 20 ms bins containing differing samples. It records the exact number of differing samples in each bin run and the peak and RMS of analyzed minus earlier samples. The three longest runs have aligned waveforms and playable excerpts covering up to their first two seconds. The two excerpts share one amplitude scale. The difference plot uses a labeled separate scale so quiet changes remain visible. The PDF records the source clocks, file hashes, times, counts, and waveforms. The JSON includes the same measurements, source ticks, checksums, and playable WAV bytes. Encoding, gain changes, and ordinary edits can produce differences. This comparison contributes no AI confidence.

## Local verification

Eight distinct retained genuine speech sources supplied three-second mono controls. Four comparisons inserted scaled recorded room audio at samples `[22400,35200)`. Four substituted known synthetic speech at `[10240,19840)`. The comparison recovered all eight exact ranges. An independent sample calculation reproduced the counts, source ticks, difference levels, and excerpt samples. All eight lossless FLAC conversions matched the original decoded values. A 0.7 gain changed all eight copies, with differences covering every bin on seven. All eight 20 ms trims were unassessed. [Source preparation and rights](audio-paired-stereo-overlap-2026-11-11.md#matched-inputs)

Authenticated app tests verify the JSON and PDF against the same decoded intervals. A second fixture preserves a nonzero 16 kHz source origin while displaying elapsed seconds relative to the first sample. Tests also verify that access checks apply and the comparison does not add objects, analyses, jobs, or payment records.

## Replay without the app

Use the project Python environment, FFmpeg, and both original files. The first command writes a directory of measurements and excerpts. The second verifies a browser JSON export, including the exact bytes of its embedded media. It also accepts a directory created by the first command.

```sh
cd backend
python -m scripts.compare_audio_references earlier.wav analyzed.wav evidence-folder
python -m scripts.compare_audio_references earlier.wav analyzed.wav chipmunk-verify-audio-comparison.json --verify
```

Replay uses the app's measurement implementation. Record the FFmpeg version if runs produce different decoded bytes. Keep private audio and exports in your local workspace.
