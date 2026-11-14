# Narrowing an existing flagged speech interval

The current short speech check displays a two-second listening range. A
600 ms edit is the shortest that range can localize at a 0.30
intersection-over-union requirement, even when it contains the entire
edit. This study tests one-second ranges inside existing flags. The
research on partial deepfake localization distinguishes evidence at a
boundary from evidence inside a segment. Our method compares the
existing classifier's local and shared-scale frame responses on the
same source clock.
[Boundary and intra-segment learning](https://arxiv.org/html/2609.25822v1)
· [Earlier shared-scale study](audio-shared-mel-context-2026-11-14.md)

## Fixed rules

Check only one-second windows wholly contained in the existing
two-second flag. Select the largest response that strictly exceeds
the corresponding development point from the parent study. A third
route shows a narrower window only when both of the separately
normalized views pick the exact same interval. Retain the broad
window otherwise. These views use the same classifier and are
correlated. The experiment measures possible changes in displayed
time. File dispositions and confidence do not participate in the
refinement.

A range is useful if it covers at least half the known edit and has
temporal intersection over union of at least 0.30. Count improvements,
lost useful ranges, narrowed ranges that miss the edit, and refinements
on already flagged untouched or resynthesized recordings.

## Previously scored recordings

| Panel | Edits with a broad flag | Useful existing | Useful after either standalone refinement | Useful after exact agreement | Wrong narrowing with exact agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Randomized stock voices | 2/16 | 1 | 1 | 1 | 1 |
| Three-generator comparison | 2/12 | 0 | 1 | 1 | 0 |
| SINE with competing speech | 1/12 | 0 | 1 | 0 | 0 |

Both dense views narrowed the same wrong stock-voice flag and the same
four ordinary comparison flags. They also narrowed one same-word
resynthesis flag. Exact agreement retained the two-second reading on a
known **1.510–1.850 s** SINE splice. The two alternatives were
**1.000–2.000 s** and **1.500–2.500 s**. Both overlap sufficiently to
count as useful. This is a retrospective comparison on the retained
parent scores.

## New readers from the publisher archive

The source freeze uses three previously unused 192 MiB prefixes at
the same pinned SINE revision. It selects one eligible utterance
from each of 15 readers absent from the previous SINE study and
preview. Each original matches the publisher-labeled cut-and-paste
version by file stem. The rule uses a 100 ms grid to select a
six-second crop containing the entire label. Edit durations range
from 230 to 790 ms. Twelve clips have edits shorter than 600 ms.

Each source pair is tested clean and mixed with the same competing
speech at −5 dB. The eight room-only controls are included. The room
recordings are reused from the previous SINE study and the new readers
use the same publisher editor. This establishes transfer to a
different set of SINE readers in the recorded source roster.
[Publisher data and labels](https://huggingface.co/datasets/PeacefulData/SINE)
· [AMI source and asset terms](audio-sine-resynthesis-transfer-2026-11-14.md#sources-and-controls)

| Delivery | Edited files flagged | Useful broad ranges | Useful after exact agreement | Ordinary files flagged | Ordinary ranges narrowed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 3/15 | 2/15 | 3/15 | 0/15 | 0 |
| Competing speech at −5 dB | 1/15 | 0/15 | 0/15 | 2/15 | 0 |

All three routes chose the same ranges on the new readers. On clean
speech they narrowed three ranges, preserved both useful broad finds
and added one useful location. The newly located splice at
**0.530–0.910 s** was shown as **0.500–1.500 s**, replacing a
**0.000–2.000 s** flag. None of the three flagged ranges in the noisy
condition qualified for refinement. Among the eight isolated rooms,
four were unassessed and four were checked without flags. All 60
foregrounds were assessed. The narrower display remains experimental
because the observed clean gain did not transfer to competing speech.

## Replay

The [public receipt](evaluation-panels/audio-narrow-flag-transfer-2026-11-14.json)
contains recorded scores and hashes of the selected inputs. The
recount and source-clock checks require only standard-library Python.
The second native preparation reproduced the freeze and all 68 PCM
commitments. It did not repeat inference.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_narrow_flag_transfer review \
  docs/evaluation-panels/audio-narrow-flag-transfer-2026-11-14.json
```

For native inference, retain the exact previous AMI background
directory and the model assets from the parent study. Acquire the same
publisher listing and bounded prefixes for shards 0, 9 and 31 using
the [SINE replay procedure](audio-sine-resynthesis-transfer-2026-11-14.md#replay).
Run freezing and scoring with a new directory and the original
matching background PCM excerpts.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_narrow_flag_transfer freeze "$run" --cache "$cache"
ln -s "$sine_panel/background" "$run/background"
backend/.venv/bin/python -B -m scripts.audio_narrow_flag_transfer score "$run"
```
