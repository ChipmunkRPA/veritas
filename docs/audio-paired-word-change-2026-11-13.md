# Compare speech word positions against an earlier recording

A local speech recognizer can supply a second timestamp when two copies
contain different words. In this comparison the recognizer also reported
word changes in ordinary noisy or reprocessed recordings. Requiring
agreement with the acoustic comparison removed those flags, but lost
useful locations on known edits. Keep the speech-recognition check
experimental and keep the acoustic comparison available separately.

## Method

Decode both recordings independently with the pinned Whisper Tiny
checkpoint, in English and with no prompt or supplied transcript.
The checkpoint card declares Apache-2.0 and the original publisher
releases its code and weights under MIT. CPU inference uses the
installed PyTorch and Transformers versions recorded in the receipt.
No new weights were downloaded.
[Pinned checkpoint](https://huggingface.co/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af)
· [Upstream license](https://github.com/openai/whisper#license)

Compare the normalized word sequences only if at least six words match
and the matching sequence covers at least 65% of the longer recording's
transcript. A candidate replacement spans one to four words on each
side. Two immediately adjacent words before and after the replacement
must match and occur only once in each copy. Word times are approximate
20 ms positions on the original 16 kHz analysis clocks. The strict
agreement arm requires overlap with the existing acoustic change on both
clocks and reports the submitted word interval. A useful location
covers half the published edit and has at least 0.30 temporal IoU.
Published research uses word-level timing and separate acoustic evidence
for related localization tasks. This experiment uses neither that model
nor its training data.
[Prior research](https://arxiv.org/html/2601.21463#S4.SS3)
· [Acoustic comparison and fixed point](audio-aligned-speech-difference-2026-11-13.md)

The 312 pairs reuse the ten SINE preview groups, 18 PartialEdit source
utterances from three VCTK speakers, and two AMI meetings. PartialEdit
retains clean, +5 dB market and −5 dB construction conditions, with a
clean reference in each case. The SINE and AMI controls include unchanged
speech, gain, filtering, resampling and an unrelated excerpt. Identical
PCM hashes share a single decode, for 302 total transcriptions.
[SINE source terms and intake](audio-sine-preview-2026-11-13.md)
· [PartialEdit source terms and mixtures](audio-partialedit-dense-transfer-2026-11-12.md)
· [AMI source terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Results

| Edited input | Pairs | Useful acoustic locations | Useful word locations | Useful word locations with acoustic agreement |
| --- | ---: | ---: | ---: | ---: |
| SINE seamless infill | 10 | 7 | 4 | 4 |
| SINE cut and paste | 10 | 6 | 4 | 3 |
| PartialEdit clean VoiceCraft | 18 | 13 | 4 | 4 |
| PartialEdit clean SSR-Speech | 18 | 0 | 0 | 0 |
| PartialEdit +5 dB market, both generators | 36 | 0 | 4 | 0 |
| PartialEdit −5 dB construction, both generators | 36 | 0 | 0 | 0 |

In the 182 ordinary pairs, the word check flagged eight of 18 market
mixtures made from genuine speech, five of 16 equalized AMI controls
and three of 16 resampled AMI controls. The acoustic check and the
agreement arm flagged zero ordinary pairs. Neither method could assess
the −5 dB speech edits. The clean SSR-Speech outputs illustrate the
difference between finding a different recognized word and finding the
annotated edit. The word check flagged 13 files but located none of the
18 known edits. On one SINE seamless example with a **4.930–5.640 s**
annotation, the acoustic range was **4.940–5.660 s** and the matching
word range was **4.920–5.460 s** on the submitted clock.

The results support using approximate word times as a secondary clue
when an earlier recording already supports a local acoustic difference.
A word mismatch by itself is unreliable in noisy or processed speech.
The stronger results are from a small number of previously studied
speakers. The test adds no serving feature or AI-confidence calibration.
[Public numerical receipt](evaluation-panels/audio-paired-word-change-2026-11-13.json)

## Replay

Use the same source caches as the acoustic comparison, including the
previously prepared market mixtures. Place the pinned checkpoint,
preprocessor and model config in `$speech`, and its tokenizer and
generation config in `$language`. The runner verifies all five hashes
before loading any weights. It can retain the decoded word lists in a
separate local file with `--private-words`; the public receipt contains
only counts, intervals and digests.
A second local run reproduced all 312 pairs byte for byte. The
portable reader returned the same counts on Python 3.11 and 3.12.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_paired_word_change score "$new_receipt" \
  --speech "$speech" --language "$language" --sine "$sine_cache" \
  --partialedit "$partialedit_run" --ami "$ami_cache"
python3 -B -S -m scripts.audio_paired_word_change review \
  docs/evaluation-panels/audio-paired-word-change-2026-11-13.json
cmp "$new_receipt" docs/evaluation-panels/audio-paired-word-change-2026-11-13.json
```
