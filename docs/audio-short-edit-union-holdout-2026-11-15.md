# Fresh holdout for combining two short speech checks

The earlier comparison recovered one extra edit when the current
two-second check and a shorter-window check were combined after seeing
the individual results. This follow-up fixes the model, two cutoffs,
candidate ranges and combination rules before scoring and tests them
on a separate source roster.
[Earlier timing comparison](audio-short-edit-duration-transfer-2026-11-15.md)

## Source selection

The SINE inputs come from eight 192 MiB shard prefixes
7, 14, 15, 16, 18, 19, 20 and 21 at the previously pinned publisher
revision. Each reader supplies one original and one labeled synthetic
cut-and-paste edit. The source roster excludes declared readers from
the earlier SINE evaluations. Files use the same six-second crop
clock within each pair. The known edit clock is used only during
source qualification and evaluation.
[SINE publisher and Apache-2.0 declaration](https://huggingface.co/datasets/PeacefulData/SINE#license)
· [Source editing method](https://arxiv.org/abs/2501.03805)

Six AMI meetings supply the room sound. Their public annotation
participants are disjoint from each other, the room-model source
roles and the preceding SINE room banks. The meetings are absent
from the evaluation docs at the reference commit. A deterministic
order selects one six-second microphone range per meeting using only
manual word timing. Eligible intervals have at least five seconds
with annotated speech and two seconds of overlapping speech. The
runner records publisher availability and picks the first six eligible
rooms it can obtain as native PCM, with a minimum of five if the
source list is exhausted. No candidate score enters enrollment.
The same room waveform and gain are used for the original and edited
member of each pair, with room sound initially set 5 dB above the
original foreground by RMS. Each isolated room is also tested.
Cross-corpus speaker identity and model pretraining exposure are
unknown.
[AMI source and CC BY 4.0 terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Operating rules

- **Current** uses the original two-second PCM check and its existing
  operating point.
- **Shorter** uses the second room model from the earlier comparison,
  its frozen width normalization for 250–1,000 ms intervals and its
  published wider point for competing speech.
- **Union** retains both flagged ranges when both checks return a result.
- **Fallback** keeps the current location when present and uses the
  shorter check otherwise.

A useful interval covers at least half the edit and 30% of the temporal
union between the prediction and the edit. The comparison counts useful
locations, original and room-only flags, unassessed files, and misplaced
spans. For an edited file, a second incorrect span counts even when
another span is useful. The matched-original comparison uses the same
edit clock and overlap rule on the genuine member of the pair.
The narrower rule is evaluated only on the prespecified mixed condition.

## Held results

The source freeze admitted 32 reader pairs and six isolated-room
controls. Each route assessed all 64 foreground deliveries. Each left
one isolated room unassessed.

| Check | Useful edited locations /32 | Edited files flagged /32 | Spans without useful edit overlap | Genuine originals flagged /32 | Isolated rooms flagged /6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 1 | 14 | 13 | 10 | 0 |
| Shorter | 0 | 6 | 6 | 6 | 1 |
| Union | 1 | 19 | 19 | 14 | 1 |
| Fallback | 1 | 19 | 18 | 14 | 1 |

Union and fallback retained the one useful current location and its
matched original stayed clear at that clock. Each added four flags
on genuine originals and one on an isolated room, with zero added
useful edit locations. The shorter rule's six ordinary flags clustered
in two room recordings, three in each. The current check flagged
every genuine original mixed with one other room, while its isolated
recording drew no flag. These recordings reuse each room waveform
across several readers, so the counts describe this mix experiment.

As a separate geometry check, the two-second grid could fit seven of
the 32 labeled edits at the required overlap. The shorter grid could
fit all 32 given the edit clock and ignoring speech admission. The
shorter rule's highest-scoring proposals overlapped only one edit
even before applying the cutoff. The ranking and acoustic response
therefore need work in addition to interval width. Both combinations
stay in evaluation. [Published measurements](evaluation-panels/audio-short-edit-union-holdout-2026-11-15.json.gz)

## Reproduce

The published receipt records model and source commitments, cutoffs,
timestamps and scalar measurements. It contains no audio, model weights
or frame features. The numerical replay requires only standard Python
and the checked-in parent receipts. A separate recount verified all
70 PCM hashes, 12 outcome rows and each paired change on two Python
versions. The narrower model run repeated byte for byte. An independent
readout reproduced the 64 source mixtures and verified the six room
controls and their annotation clocks. Recomputing the two-second
selection outside the app's selection function reproduced 18 decisions
covering one reader pair and the isolated control in every room.

```sh
export PYTHONPATH=backend
module=scripts.audio_short_edit_union_holdout
receipt=docs/evaluation-panels/audio-short-edit-union-holdout-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

For native replay, download the eight exact prefixes using the existing
[bounded download procedure](audio-sine-resynthesis-transfer-2026-11-14.md#replay).
Use the pinned AMI manual annotations and the original
[room model fit](audio-short-edit-room-training-2026-11-15.md#reproduce-locally).
The freeze uses the repository history at `63eda8e9` for the meeting
exclusion list. It records which publisher requests yielded PCM. To
repeat on the exact acquired files, retain the original frozen plan and
its hash-named `pcm/` directory for the scoring commands.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$run/plan" "$cache" "$ami_annotations"
"$runner" -B -m "$module" current "$run/plan" "$voice_weights" "$run/current.json"
"$runner" -B -m "$module" shorter "$run/plan" "$voice_weights" "$run/shorter.json" "$room_fit"
python3 -B -S -m "$module" combine "$run/plan" "$run/current.json" "$run/shorter.json" "$run/receipt.json.gz"
```
