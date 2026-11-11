# Blind English transcript comparison on the GRID pilot

This lab study compares the eight original GRID waveforms that matched a
publisher word reference in v162. All sixteen roster rows remain in the same
order, including the eight unmatched rows. The v162 freeze, private input and
observation commitments, and actual public and local result must agree. The
whole original mono 16 kHz WAV is verified against its v162 SHA-256 before
conversion from signed 16-bit samples to float32. The original sample count,
values, and origin are preserved. [Prior result](grid-phone-alignment-v162-result.md)

The CPU feature extractor uses the existing v110 whole-waveform log-Mel
settings. The complete multilingual Whisper Tiny encoder, decoder, and tied
projection use the existing safetensors and the same publisher revision as
v110. The published Hugging Face artifact declares Apache 2.0. The revision's
`tokenizer.json` and `generation_config.json` require exact byte counts, Git
blob OIDs from the pinned publisher tree, and whole-file SHA-256. The runner
checks these cached artifacts offline. The OpenAI Whisper source repository
has its own MIT license. [Publisher revision](https://huggingface.co/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af) · [Publisher tree metadata](https://huggingface.co/api/models/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af) · [Source license](https://github.com/openai/whisper/blob/main/LICENSE)

Before inference, freeze the installed runtime and decoding settings. Each
available waveform receives one greedy English transcription with one beam,
temperature zero, and a 64-new-token maximum. Keep the publisher's token
suppression, fixed English transcription prefix, and no-timestamp mode.
There is no supplied prompt, grammar, reference text or word boundary, token
conditioning from an earlier clip, or fallback temperature. Discard an
unterminated or nonconforming decode as unavailable. The decoder receives the
waveform alone. Publisher words enter only after blind transcription.

Score exact word sequences and unit-cost word edit distance. The primary
score lowercases output and extracts word tokens, retaining numbers as
written. The fixed secondary score maps isolated ASCII digits to their
English spellings. Empty completed hypotheses count as deletions. Report
paired v162 outcomes on the same assessed originals and retain unavailable
rows. Only aggregate counts and whole-artifact commitments may be published.
Hypotheses, source paths, reference text, and observations stay local. A
second pass with forced alignment would require its own frozen comparison and
independent phone timing references. This study supports no inference about
achieved lip alignment or production confidence.

Run from `backend` with the existing local environment. Use `python -m
scripts.grid_blind_transcript_v163 freeze` to bind the code, inputs, assets,
and method before using `python -m scripts.grid_blind_transcript_v163 run`.
The latter uses the existing v108 authored neural guard and never retries or
overwrites an attempted stage. The guard writes its profile after the worker
returns. Review it together with the aggregate result before publication.
