# Align two recordings to locate changed speech

The earlier-copy check located seven of ten seamless SINE edits where the
current app comparison located none. It also located 13 of 18 clean
VoiceCraft edits on a second dataset. It declined every −5 dB construction
mixture and every clean SSR-Speech edit in that test. A follow-up
processing check removed all nine resampling and equalization flags
on the same SINE controls. The method remains experimental and
requires an earlier copy of the same recording.

## Method and inputs

Dynamic time warping lets the two copies have locally different timing.
Compute 12 gain-normalized cepstral coefficients and their differences
on 40 ms frames every 20 ms. Align the resulting features within 800 ms
of the duration-adjusted diagonal. Average the path distances on the
submitted timeline and search for a higher-distance interval flanked by
400 ms of similar active sound on each side. The best interval can range
from 120 ms to 2.4 s. The measurement uses only the two waveforms. Recent
work uses the same general alignment idea to compare a waveform to a
watermark reconstruction. This comparison instead takes a supplied
earlier copy. It uses no paper code, watermark or model weights.
[Reference-alignment method](https://arxiv.org/html/2608.25289v1#S2.SS2)

The fixed point is the largest eligible control response, 0.443620, on
80 original, gain, equalized, resampled and wrong-excerpt pairs from
16 excerpts in two AMI meetings. The point was recorded before either
edit bank was processed. An edit is useful if it covers at least half
the labeled range and reaches 0.30 temporal intersection over union.
The current comparator runs the existing earlier-copy route with timing
and clean-speech checks enabled. It tests every reported range, using
the submitted file's own clock, including after trim alignment.

The original SINE audio is 24 kHz mono PCM16. The runner verifies the
publisher files and uses one deterministic 24-to-16 kHz resample. The
ten originals come from one reader and each has a cut-and-paste and a
seamless version from the same publisher generator. The seamless
versions also reconstruct the surrounding speech. PartialEdit supplies
18 English utterances from three VCTK speakers with both VoiceCraft and
SSR-Speech edits. It uses the previously verified clean 16 kHz samples
and the existing −5 dB construction mixtures. The earlier reference
remains clean in both delivery conditions. These are new measurements on
previously studied sources. No generalization to a new speaker or
generator population follows from this test.
[SINE construction and dataset terms](audio-sine-preview-2026-11-13.md)
· [PartialEdit inputs and terms](audio-partialedit-dense-transfer-2026-11-12.md)
· [AMI source terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Results

| Edited input | Pairs | Assessed by alignment | Useful current ranges | Useful aligned ranges |
| --- | ---: | ---: | ---: | ---: |
| SINE seamless infill | 10 | 10 | 0 | 7 |
| SINE cut and paste | 10 | 10 | 2 | 6 |
| PartialEdit clean VoiceCraft | 18 | 15 | 0 | 13 |
| PartialEdit clean SSR-Speech | 18 | 0 | 0 | 0 |
| PartialEdit −5 dB, both generators | 36 | 0 | 0 | 0 |

For a SINE infill labeled **4.930–5.640 s**, the selected submitted
range was **4.940–5.660 s** and its aligned earlier range was
**4.920–5.540 s**. The alignment produced one misplaced range in
each of the SINE cut-and-paste and clean VoiceCraft groups. It
withheld all ten unrelated SINE excerpts from the same reader, all
18 noisy genuine counterparts, and flagged none of the untouched or
gain-adjusted SINE controls.

After seeing that three equalized and six resampled SINE controls
flagged, a follow-up tested whether a single linear filter and small
uniform clock change could explain both files. It estimated the
filter on alternating blocks and checked the other blocks, then
reversed the split. Direct approximation of the ratio of sample counts
left three resampling flags because the renderer rounds its output
length up. The final guard instead tries the simplest rational ratio
whose rounded output length matches the submitted clock. It kept the
original alignment point and all 26 useful edit ranges, and removed all
three equalization and six resampling flags. This is a follow-up on the
same SINE controls. Two separate AMI meetings supplied 80 further
controls with a different resampling ratio. All 16 unrelated excerpts
were declined and the other 64 pairs produced zero local flags with
either alignment arm. These additional controls use the same types
of ordinary processing.

The clean result supports further earlier-copy testing. A fresh test
needs separate source speakers and generators, room and codec changes,
timing drift, and ordinary local speech edits. The alignment found
no ranges under strong noise and cannot support a standalone AI
classification. No app route or AI-confidence score uses these results.
[Scalar evidence and counts](evaluation-panels/audio-aligned-speech-difference-2026-11-13.json)

## Replay

Follow the [pinned SINE download](audio-sine-preview-2026-11-13.md#replay)
and the [AMI cache acquisition](audio-bandwidth-return-2026-11-13.md#replay-from-public-source-ranges).
Use the prepared PartialEdit PCM directory from its earlier test, or
rebuild it with that test's pinned publisher ranges and backgrounds.
Each WAV and prepared PCM input is checked before use. The standard-library
reviewer checks the method, planned rows, native ranges and counts. The
numerical runner recalculates every result without model inference. A
second local run reproduced all 338 pairs byte for byte.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 -B -S -m scripts.audio_aligned_speech_difference review \
  docs/evaluation-panels/audio-aligned-speech-difference-2026-11-13.json
backend/.venv/bin/python -B -m scripts.audio_aligned_speech_difference score "$new_receipt" \
  --sine "$sine_cache" --partialedit "$partialedit_run" --ami "$ami_cache"
cmp "$new_receipt" docs/evaluation-panels/audio-aligned-speech-difference-2026-11-13.json
```
