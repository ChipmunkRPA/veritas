# Find exact repeats in a stereo upload

Chipmunk Verify checks the left and right channels separately on new analyses of supported audio files and video soundtracks. A repeated section can remain in one channel after an edit while the other channel carries different sound. Mixing the two channels can hide the exact match. The check reports both occurrences as sample intervals and original source timestamps. Media Evidence plays both occurrences. The browser, Standard PDF and JSON include the shared waveform and exact source positions. This check identifies repeated samples. Ordinary edits, repeated alerts and loops can produce this pattern. It carries zero weight in the AI assessment. [Related audio forensics research](https://arxiv.org/abs/2302.07584)

The first audio stream must contain two ordered channels and a continuous native 16 kHz sample clock. The analyzed track must contain 2.5–4 seconds of signed 16-bit PCM or 16-bit FLAC. The check reuses the bounded exact-sample search on each channel with a 150 ms minimum match. Clips with insufficient variation or too many possible repeats report that the search could not establish a result. Other input profiles do not receive this extra check. The original mono repeat check and voice scoring continue separately.

## Local evidence

Sixteen previously used recordings from two room panels supplied competing speech and background-only controls. Each provided a fixed 49,501-sample stereo excerpt. Each excerpt was delivered unchanged and with a separate exact copy in either channel. The opposite channel was left byte-for-byte intact. All 32 authored edits were localized at the exact boundaries in the selected channel. The downmix found none of them. All 16 unchanged controls had zero reported matches. The source PCM and the earlier room rosters were verified before rendering. These are controlled repetitions on previously used recordings. They do not establish sensitivity to remixed, resampled or lossy material. [Room and source qualification](audio-measured-spatial-transfer-2026-11-12.md#sources-and-fixed-comparison)

Local prepaid-flow verification used a matching ordinary stereo WAV and its left-channel edit, with simulated Stripe. Both queued only after payment. The positive evidence showed the exact two source ranges and the negative had no channel matches. Chrome played both occurrences and downloaded the same JSON and PDF as the API. A signed package independently re-decoded the source and verified the repeat clips. Separate audio and video regression cases check exact intervals and original-channel playback, a non-whole-second audio tail, source timestamp offsets, and rejection of changed signed coordinates. The source and test media remain local.

## Verify a signed export

From a clean checkout with FFmpeg and the backend verification environment installed, use an export and the original file. For a video, pass the original video as `--source-audio` to select this verification, and optionally add `--source-video` for picture evidence.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m app.cli verify "$package" \
  --public-key "$trusted_key" --source-audio "$original" --json
```

The ordinary signed-package verifier checks the declared WAV asset and sample hashes without loading a model. Supplying the original enables native channel decode and repeats the exact interval search for both channels. The report never embeds the complete stereo track.
