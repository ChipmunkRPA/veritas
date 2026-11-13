# Rebuild the matched speech test from public sources

The original ODSS study compared native mono with two alternate views of
genuine and synthetic speech under overlapping voices. The public result
already supports replaying its decision arithmetic. The new command
reconstructs the exact submitted stereo waveforms and the classifier input
windows directly from their publishers. A new reviewer can start with an
empty cache. [Original matched-room study and scalar receipt](audio-odss-matched-room-2026-11-13.md)

## Source bytes

- The ODSS range requests cover the ZIP directory, the required metadata and
  48 utterance WAV members. The runner verifies the directory SHA-256 and
  archive-member headers, lengths and CRCs. The source roster must match the
  original sealed commitment. [ODSS release and CC BY-SA 4.0 terms](https://zenodo.org/records/8370669)
- The AMI annotation archive is checksum-pinned. The runner independently
  selects the same eight six-second intervals per meeting from the published
  word timings. It fetches only the two exact RIFF headers and 16 mono PCM
  excerpts from the original WAV URLs. Each header must match its recorded
  checksum, signed PCM format and file length. The exact PCM slices are
  checked against the previously sealed background roster. The quoted
  whole-file checksums identify the original parent files. Bounded retrieval
  verifies the selected ranges rather than computing full parent WAV hashes.
  [AMI corpus downloads](https://groups.inf.ed.ac.uk/ami/download/)
  · [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
- The Surrey and BBC room responses are downloaded directly from their
  source repositories and verified against the original Git blob commitments.
  [Surrey measured dataset and license](https://doi.org/10.15126/surreydata.00813511)
  · [BBC static response attribution and terms](audio-bbc-room-assets-v303-method.md)

From an empty directory, the verified download used **47,922,098 bytes**
across 75 cache files. The runner matched all **176 stereo deliveries**,
including **128 noisy or background-only deliveries**, and all **432
comparison-view slots** against the original source-window commitments.
It also matched the per-delivery SNR, acoustic gain, and FFT comparison
fields. [Public replay receipt](evaluation-panels/audio-odss-public-waveform-replay-2026-11-13.json)

The cache contains the selected source recordings and excerpts, so it belongs
outside the repository. The public result records counts and checksums. The
waveform run makes no classifier calls. The separate scalar reader checks
the earlier decision arithmetic. Neither replay supplies a new accuracy
estimate.

## Reproduce

Use the backend numerical dependencies and an empty scratch directory.
`acquire` retrieves the public inputs and checks the waveform commitments.
`replay` does the same waveform checks from a populated cache without
network access.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.replay_audio_odss_public acquire "$cache"
backend/.venv/bin/python -B -m scripts.replay_audio_odss_public replay "$cache"
python3 -B -IS backend/scripts/review_audio_odss_matched_room.py \
  docs/evaluation-panels/audio-odss-matched-room-2026-11-13.json
```
