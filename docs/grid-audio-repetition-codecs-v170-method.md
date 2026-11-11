# Repeated-waveform cues after ordinary audio encoding

The exact PCM cue located the surviving copy in all edited GRID diagnostic
cases. This study checks how much of that evidence remains after encoding.
It compares the same exact cue with an independently specified waveform
similarity check. Both outputs describe paired sample ranges. Neither
supplies a speech-origin vote.
[Previous result](grid-audio-splice-probe-v167-result.md) · [Audio copy-move research](https://arxiv.org/abs/2302.07584)

Use the same 16 consumed genuine pilot recordings and all four frozen
variants per recording. Each variant enters four channels: unchanged
PCM, AAC at 64 kb/s in MP4, MP3 at 64 kb/s, and Opus at 48 kb/s in Ogg.
This gives 256 planned observations. Codec choices and cue code are
fixed before replay. No source is selected by voice score or cue output.

Verify every encoded asset with the production mono decoder. Require one
continuous source clock and the observed codec stream origin. MP3 may
carry a nonzero container origin after gapless decode. AAC decoder padding
beyond its exact declared track duration is recorded and excluded. Other
decoded lengths must match the input length. Evaluate sample intervals
relative to the first decoded sample. Retain failed channel observations.

The adjudication counts a witness only when both reported ranges overlap
the surviving donor and copied output by at least 150 ms. Exact matching
requires the authored lag. Similarity permits an error of at most 32
samples, or 2 ms. Retain the number of untouched cases with any reported
match, all ambiguous and unavailable outcomes, and the exact source
denominators. The encoder experiment can measure repeat coverage on the
known ordinary copy. It leaves precision on fresh genuine or AI speech
and mixed or noisy channels unmeasured.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.grid_audio_repetition_codecs_v170 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.grid_audio_repetition_codecs_v170 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.grid_audio_repetition_codecs_v170 readout
```

Use the existing data-only resource monitor. No model inference runs in
this study. Per-case sample coordinates and codec outputs stay private.
