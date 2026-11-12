# Longer clips with newly constructed voice edits and recorded room speech

The earlier meeting-audio edge gain did not transfer. The edge check added zero useful locations among nine held edits at its frozen point and found none at up to two false genuine groups. The current check found two at one false group in the retrospective ranking. The app continues to use its existing voice check. [Aggregate results and commitments](evaluation-panels/audio-spectra-long-room-holdout-2026-11-12.json)

## Prospective panel

The earlier overlapping-context check located three of six held meeting-audio edits at two retrospective false alarms. It missed all six at the frozen point. This trial asks whether the signal transfers to new placements, longer clips and separate rooms. The run keeps the score and the exclusive-edge rule fixed. It independently measures the existing voice model on actual two-second contexts of the new recordings. [Earlier context trial](audio-spectra-context-transfer-2026-11-12.md) · [Localizing edits in longer multi-speaker audio](https://arxiv.org/abs/2609.10051)

Freeze 43 exact inputs before inference. Fourteen genuine controls from six splice episodes and two isolated room recordings set the operating points. The held panel contains nine known 0.8-second AI edits, 18 matched genuine splices, and two isolated room recordings. Each held episode uses one of nine distinct genuine host groups and one of three independent donor groups from each of clone, expressive and stock synthesis. The donor groups exclude all groups in the earlier Spectra meeting-audio panel. The point and held roles use distinct qualified LibriSpeech source groups and different AMI meetings. All 16 recorded meeting participants are distinct, and the held rooms differ from the earlier Spectra meeting rooms. [LibriSpeech rights](https://www.openslr.org/12/) · [AMI Corpus terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) · [Qualified generator source terms](audio-short-edit-source-plan-v169.md)

Use six or eight seconds from the original speech after its previously analyzed four-second prefix. Insert an eligible same-host snippet, a different genuine voice, or a generated donor at the same fixed location, with matched 20 ms fades and RMS. Render the same AMI capture behind each matched trio, 5 dB louder than the unedited foreground over the full clip. AMI cuts have more than 80% annotated speech and at least 1.2 seconds of simultaneous annotated words. The new recorded intervals exclude exact windows recorded in the archived native, harmonic and spatial qualification receipts. All source banks and meeting recordings had prior project exposure. The exact long cuts, mixtures and edit locations are new, and pretraining exposure remains unknown. [Source qualification](audio-complete-sources-v81-result.md) · [Runner](../backend/scripts/audio_spectra_long_room_holdout.py)

## Fixed comparison

For the independent model, score native three-second contexts with starts one second apart. The maximum raw response retains a three-second window. The largest difference between consecutive windows nominates only the one-second edge unique to the stronger context. The two shared seconds remain unresolved. For the incumbent, score two-second contexts starting at 1 and 3 seconds, with a third at 5 seconds on eight-second files. The ordinary speech activity gate must admit each incumbent window. No training or customer calibration takes place. The archived independent model and the current weights are pinned to their existing audited hashes. [Independent weights and preprocessing](audio-independent-background-transfer-2026-11-11.md) · [Current model audit](commercial-model-audit.md)

New fixed points strictly exceed the highest response among the 14 genuine point controls. Report the previous points separately. The union keeps any flag and its proposed interval from either current or edge check. A useful interval must reach 0.30 intersection over union and 0.50 edit coverage. Retrospective equal-false-alarm comparisons rank with held controls and cannot establish a service operating point. The advancement criterion asks for two added useful locations from different generator families and both held rooms, at most one additional false group, and no flagged isolated room.

## Held results

| Frozen comparison on nine edits and 20 genuine controls | AI flags | Useful edit locations | False genuine groups |
| --- | ---: | ---: | ---: |
| Existing two-second check | 0 | 0 | 0 |
| Independent raw maximum | 1 | 0 | 2 |
| Exclusive one-second edge | 0 | 0 | 0 |
| Existing check or exclusive edge | 0 | 0 | 0 |

| Retrospective false groups | Existing two-second useful locations | Independent raw maximum useful locations | Exclusive-edge useful locations |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0 | 0 |
| 1 | 2 | 0 | 0 |
| 2 | 2 | 0 | 0 |

All three checks also returned zero useful locations and zero false groups at their previous frozen points. The independent maximum detected one AI edit at the new point but its three-second range was too broad to count as a useful location. The two resulting false groups appeared in the same held room, one on six-second and one on eight-second audio. The existing model left one held genuine clip unassessed. The edge did not add a useful location or falsely flag an isolated room recording.

Every edit was geometrically reachable by some one-second edge, including all five six-second and four eight-second inserts. Before applying a point, the edge check placed a useful range on three edits and the current check on two. The independent maximum and the edge each ranked five of nine synthetic clips above both matched genuine controls. The current check did so on two. That matched-pair result did not translate into a better thresholded ranking across genuine source groups. Eight exact edit intervals contained some annotated interference, and three contained overlapping words from at least two background speakers. Annotated interference occupied 6.33 of 7.20 edit seconds, including 1.74 seconds of overlapping background words. The −5 dB level describes the full waveform rather than each edit interval.

## Replay

The CPU run made 212 independent-model forwards and 102 admitted current-model forwards. The runner independently regenerated all 43 original matched mixtures and controls. The standard-library reviewer rehashed each waveform, rebuilt every graph input, checked sample identity outside matched edit intervals, and replayed fixed and equal-alarm decisions. One setup attempt ended after the independent point controls when the incumbent's Torch runtime received a duplicate inter-op setting. The corrected attempt reproduced the same source roster and exact independent point responses, and froze all points before any held scoring. Use the retained Spectra asset directory as `weights` and put a compatible CPU ONNX Runtime on the Python path. The two model stages run sequentially.

```sh
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_spectra_long_room_holdout freeze "$run" "$weights"
for role in selection confirmation; do
  for arm in spectra current; do
    backend/.venv/bin/python -B -m scripts.audio_spectra_long_room_holdout score "$run" "$weights" --role "$role" --arm "$arm" || exit 1
  done
  if [ "$role" = selection ]; then
    backend/.venv/bin/python -B -m scripts.audio_spectra_long_room_holdout points "$run" "$weights" || exit 1
  fi
done
for phase in report audit; do
  backend/.venv/bin/python -B -m scripts.audio_spectra_long_room_holdout "$phase" "$run" "$weights" || exit 1
done
python3 -S backend/scripts/review_audio_spectra_long_room_holdout.py "$run" "$annotations"
```
