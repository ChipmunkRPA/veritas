# Local speech edit comparison on the SINE preview

The public SINE v1 preview supplies all three versions of ten matched
English audiobook excerpts. Each has an original recording, a seamless
infilling edit, and a synthesized replacement inserted by cut and paste.
The authors used their own Voicebox reimplementation for both edited
versions. The infilled version also reconstructs surrounding speech, so a
file-level detector could respond to processing outside the edited word.
The preview supplies no resynthesized-genuine comparator or independent
generator. [Dataset card](https://huggingface.co/datasets/PeacefulData/SINE) ·
[Publisher construction paper](https://arxiv.org/html/2501.03805#S3.SS2)

The pinned publisher revision declares Apache 2.0. LibriVox releases
the underlying US recordings into the public domain. We used the preview
locally for product evaluation and retained numeric receipts in the
repository. The preview is ungated and the run used no additional archives,
external inference, generator code, or customer data.
[Publisher license metadata](https://huggingface.co/api/datasets/PeacefulData/SINE/revision/81be5d5e6a7d5d0f9bbbb613abed7b700f43d793) ·
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) ·
[LibriVox public-domain terms](https://librivox.org/pages/public-domain/)

## Verified input clocks

The preview's 30 WAVs total 10,305,560 bytes. Each publisher size and LFS
SHA-256 matches its downloaded file. All files decode as mono signed
16-bit PCM at 24 kHz. Each pair of edited variants agrees on the
published interval and native sample count. Final annotations differ from
the decoded lengths by −80 to +112 samples, at most 4.67 ms, consistent
with timestamps rounded to the nearest centisecond. The receipt
retains this difference. Internal boundaries come from the
publisher annotations at that precision. One LibriVox reader contributes
all ten excerpts across three chapters of the same recording. Across
the ten groups, edited intervals last from 0.11 to 0.71 seconds.
[Native clocks, identities and hashes](evaluation-panels/audio-sine-preview-2026-11-13.json)

## Frozen localization comparison

The same checkpoint scores the two-second windows at 1 s and 3 s and
the experimental one-second intervals on a half-second grid. The latter
uses the existing first and last four-second context method. The lower
frame-score median is used if both contexts cover a window. Both methods
use the same speech activity gate and their previously fixed operating
points. The production decoder converts 24 kHz to 16 kHz and verifies a
continuous source clock starting at zero. Each location must cover at
least half of its annotated edit and reach 0.30 intersection over union.
Every supplied edit remains in the comparison, including edits too short
for either window to reach that criterion.
[Frozen comparison and prior data](audio-source-isolation-diagnostic-2026-11-13.md)

| Case type | Two-second flags | One-second flags | Two-second useful locations | One-second useful locations |
| --- | ---: | ---: | ---: | ---: |
| Original | 0/10 | 0/10 | — | — |
| Seamless infilling | 9/10 | 10/10 | 0/10 | 0/10 |
| Cut and paste | 1/10 | 4/10 | 0/10 | 2/10 |

Both methods scored every file. The one-second grid could reach nine
annotations in each edited group, while the two-second windows could
reach zero. Before the operating point was applied, the strongest
one-second windows overlapped seven of ten cut-and-paste edits
sufficiently and zero of ten infills. The one-second method ranked all
20 manipulated versions above their own original. The two-second method
did so for ten infills and seven cut-and-paste cases.
[Scalar scores, points, and exact selected intervals](evaluation-panels/audio-sine-preview-transfer-2026-11-13.json)

For the same annotated change at 4.93–5.64 s, the one-second method
selected 5.00–6.00 s on the cut-and-paste edit and 6.00–7.00 s on the
infilling edit. The latter misses the edit entirely. All ten infills
produced a file-level flag, and none produced a useful location. This
is consistent with a detector response to reconstruction elsewhere in
the file. Ten genuine excerpts from one reader give only a narrow
false-alarm check. The run used no added background audio and does not
establish performance under noise. It made no serving or calibration
changes. A second local run reproduced the scalar receipt byte for byte.

## Replay

With the publisher listing and verified WAVs in a local scratch directory,
the input audit runs using only the Python standard library. Fetch the
listing and CSV at the pinned revision, and each WAV through its exact
listed path. Keep the input WAVs outside the repository.

```sh
set -e
scratch=$(mktemp -d)
rev=81be5d5e6a7d5d0f9bbbb613abed7b700f43d793
base=https://huggingface.co/datasets/PeacefulData/SINE
curl -fLsS "https://huggingface.co/api/datasets/PeacefulData/SINE/tree/$rev/preview/audio" \
  -o "$scratch/sine-preview-audio.json"
curl -fLsS --max-filesize 6725 "$base/resolve/$rev/preview/metadata.csv" \
  -o "$scratch/sine-preview-metadata.csv"
jq -r '.[] | [.path, .size] | @tsv' "$scratch/sine-preview-audio.json" |
  while IFS="$(printf '\t')" read -r path size; do
    curl -fLsS --max-filesize "$size" "$base/resolve/$rev/$path" -o "$scratch/${path##*/}"
  done
python3 -B -IS backend/scripts/review_audio_sine_preview.py "$scratch" \
  --compare docs/evaluation-panels/audio-sine-preview-2026-11-13.json
```

Local scoring and scalar replay use the same files as the input audit.
Scoring uses the pinned local voice weights and the existing offline
evaluation environment.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_sine_preview_transfer score \
  "$scratch/scored.json" --sources "$scratch"
python3 -B -S -m scripts.audio_sine_preview_transfer review \
  docs/evaluation-panels/audio-sine-preview-transfer-2026-11-13.json
```
