# New synthetic Turkish sources for background-noise testing

Select 16 previously unused Alania clips from the exact licensed shard
used for the earlier transfer study. The catalog filter keeps 3–4 second
publisher-filtered recordings from designed voices and CC BY 4.0 domain
text. Exclude all 24 previously selected row IDs, row locations, voice IDs
and source document IDs. Select one clip per remaining designed voice and
per source document in the existing deterministic hash order. Freeze the
plan before decoding the selected waveforms.
[Pinned dataset and license](https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr/tree/277dfc8a2322014099ff72688bcaee12480c4605) · [Earlier roster](audio-alania-roster-v178-method.md)

Decode with the same bounded FFmpeg path as the earlier study. Preserve
the complete native source extent and exact 16 kHz sample mapping. A clip
shorter than three seconds after decoding is unavailable. Check exact
source, waveform and decoded PCM duplicates against the 24 prior cases
and the other selected clips. Retain unavailable slots with their reasons
and keep row identities, media and per-source hashes private. Publish
aggregate counts and artifact commitments. This is a new source cohort
within one known generator family. A second generator and matched genuine
speech are prepared separately. This preparation makes zero model calls.
[Exact preparation method](audio-alania-transfer-v179-method.md)

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_alania_fresh_v213 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_alania_fresh_v213 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_alania_fresh_v213 prepare
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_alania_fresh_v213 readout
```
