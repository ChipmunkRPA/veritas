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

## Gain and polarity

Set `VERITAS_AUDIO_STEREO_GAIN_REPEAT_REVIEW=true` in local development to also check for repeated excerpts at a different level or inverted polarity. It uses the same supported native stereo inputs and 150 ms minimum as the exact repeat check. Each candidate must reproduce the other excerpt sample for sample after applying one tested rational multiplier and rounding to 16-bit precision. The recorded relationship can run in either chronological direction. It does not establish edit order or authorship.

The ordinary evidence view and Standard PDF show the ratio, two source ranges and two waveforms drawn at the same amplitude scale. The evidence view and JSON link both original WAV excerpts. The signed package verifier recomputes their exact gain relationship, and supplying the original recording or video also replays the channel search and both source excerpts. Matching silence may extend a range beyond the active event. A loop, alert or other ordinary repeat can satisfy the check. Mixing separate noise into the same channel, clipping, resampling and lossy encoding can defeat it. The result carries zero AI confidence weight.

The [frozen transfer and replay](audio-gain-repetition-transfer-2026-11-13.md) found 96 of 96 authored gain-related repeats and zero flags on 144 matched controls across 24 new excerpts from two previously used AMI meetings. Local Chrome loaded both WAV intervals with the expected playback duration, rendered both waveforms and downloaded byte-identical JSON and PDF. Three isolated analysis runs each queued only after a simulated $9.99 USD payment. The original-source tests covered a WAV and an offset video soundtrack, including rejection of a changed gain ratio in a re-signed package.
