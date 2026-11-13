# Bounded native Half-Truth speech intake

The public Half-Truth Audio Detection v1 release contains Mandarin recordings
with short splices authored by the benchmark publishers. Its paper identifies
GST-Tacotron with an LPCNet vocoder as the synthesis pipeline for the partial
set. The publisher's release metadata declares CC BY 4.0 and its base
AISHELL-3 release declares Apache 2.0. This check uses the published audio
only and requires no generator code, checkpoint or remote inference.
[HAD data record](https://zenodo.org/records/10377492) ·
[License metadata](https://zenodo.org/api/records/10377492) ·
[Author's paper](https://arxiv.org/abs/2104.03617) ·
[AISHELL-3 terms](https://www.openslr.org/93/)

Three HTTP range requests fetched 4,340,498 bytes from the 8,073,665,280-byte
ZIP64 archive. The ZIP tail located the small label and readme members and
the separately deflated, nested test archive. Its first 4 MiB, starting at
the outer member header, yielded 41 complete WAVs and the start of file 42.
The nested test archive is compressed as an outer member. Randomly seeking
to an inner ZIP offset without an outer DEFLATE index would read the wrong
bytes. The local verifier pins each range, checks exact lengths and
Content-Range, validates both complete metadata member CRCs, then validates
each completed inner WAV's CRC and native PCM length.

| Measurement | Count |
| --- | ---: |
| Publisher test labels read | 9,072 |
| Partially replaced test clips | 8,838 |
| Fully synthetic test clips | 234 |
| Fully genuine test clips | 0 |
| Complete native WAVs in the bounded prefix | 41 |
| Native 16 kHz mono PCM16 clocks verified | 41 |
| Score-blind, short interior splice matches | 8 |

The intake requires a four- to nine-second file, one 0.50–1.20-second
replaced interval and at least 0.40 seconds of original audio on either
side. Publisher endpoints are centisecond labels mapped to the native
16 kHz clock. They do not establish exact sub-centisecond cut positions.
The final annotation precedes the end of the native waveform by 0–144
samples. The verifier retains that tail separately and does not label it.

The test labels include no source-speaker ID, parent utterance ID or
per-clip donor ID. The eight qualifying clips form a positive-only
Mandarin transfer for the named synthesis pipeline. They can measure
location recovery and abstentions on these clips. They cannot estimate a
false-alarm rate, establish speaker-independent performance within the
corpus or confirm performance on English or newer voice generators. The
per-file commitments in the
[numeric receipt](evaluation-panels/audio-had-native-intake-2026-11-13.json)
include no audio or transcripts.

## Positive-only localization check

The incumbent checkpoint scored all eight native, unmodified inputs after
the source receipt was frozen. The existing two-second stock comparison
uses windows at seconds 1 and 3 when they fit. The dense experimental
comparison uses a one-second window at every half second, with source-frame
scores read from the first and last four-second contexts. Both retain the
same speech activity gate and their earlier fixed points. The comparison
uses the strongest scored window per clip and requires at least 0.30
intersection over union and half of the published splice covered.
[Earlier calibration and controls](audio-source-isolation-diagnostic-2026-11-13.md)

| Native route | Assessed | Flagged | Reachable by window geometry | Useful location | Misplaced location |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock two-second windows | 8/8 | 8/8 | 2/8 | 2/8 | 6/8 |
| Dense one-second windows | 8/8 | 8/8 | 8/8 | 8/8 | 0/8 |

The stock route selected 1.000–3.000 s on every case. The six misplaced
windows could not satisfy the overlap criterion on that route even before
scoring. For example, the annotation on one file marks 2.480–3.000 s. The
stock range covers it but spreads the finding over two seconds. The dense
check shows 2.500–3.500 s and clears both location criteria. The result
shows that shorter, more frequent windows recovered location on these
positives. All eight files were already detected by either route. Genuine
controls are still required before the dense check could affect an AI
verdict or be interpreted as a general detection improvement. No files in
this test include new injected noise. [Scalar result and exact windows](evaluation-panels/audio-had-positive-transfer-2026-11-13.json)

## Replay the intake

Use these exact public publisher byte ranges from a clean checkout. The
script runs with the Python standard library and reports the same receipt
on Python 3.11 and 3.12. To keep eligible WAVs privately for scoring, pass
`--media-dir` with a new scratch directory.

```sh
set -e
scratch=$(mktemp -d)
base='https://zenodo.org/records/10377492/files/HAD.zip?download=1'
curl -fsSL --range '8073599723-8073665279' --max-filesize 65557 \
  -D "$scratch/tail.headers" "$base&chipmunk_part=zip_tail" -o "$scratch/tail.bin"
curl -fsSL --range '1719034985-1719115621' --max-filesize 80637 \
  -D "$scratch/testlabels.headers" "$base&chipmunk_part=testlabels" -o "$scratch/testlabels.bin"
curl -fsSL --range '1719115622-1723309925' --max-filesize 4194304 \
  -D "$scratch/testprefix.headers" "$base&chipmunk_part=testprefix" -o "$scratch/testprefix.bin"
python3 -B -IS backend/scripts/review_audio_had_native_intake.py "$scratch" \
  --compare docs/evaluation-panels/audio-had-native-intake-2026-11-13.json \
  --media-dir "$scratch/eligible"
```

Native scoring uses the pinned voice weights and local environment. The
second command recounts the supplied scalar receipt using Python's
standard library and does not load the checkpoint.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_had_positive_transfer score \
  "$scratch/scored.json" --sources "$scratch/eligible"
python3 -B -S -m scripts.audio_had_positive_transfer review \
  docs/evaluation-panels/audio-had-positive-transfer-2026-11-13.json
```
