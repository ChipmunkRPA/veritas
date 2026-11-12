# Held check of marked video timing fallback

Keeping every assessed result from the production comparison and adding the short-template fallback localized **11 of 12** native shifts, up from **9 of 12**. The two gains came from separate actor groups. All nine existing locations were preserved. Neither arm alerted on the 12 unchanged and gain-only controls. [Held counts and verification](evaluation-panels/grid-marked-release-holdout-2026-11-12.json)

## Frozen held study

Use six GRID actor groups outside the previous marked-timing transfer. Select two previously unplanned original utterances per group in hash order from the rights-qualified native archive. Each target contains /p/ near a published word onset. The actor identities have prior project exposure. The selected exact files and word intervals were frozen before looking at mouth geometry. The first candidate from each group had a supported last-closed native picture on the muted video, with room for both planned sound shifts. No acoustic landmark was used to choose the mark or define the edited range. [GRID publisher data](https://zenodo.org/records/3625687) · [Earlier transfer and template parameters](grid-marked-release-transfer-2026-11-12.md)

Each marked original supplies four deliveries. One is unchanged, one changes gain within a fixed local window, and two move exact audio samples 240 ms earlier or later within that window. Actor groups alternate between 600 ms and 760 ms edited windows. All native pictures and the samples outside the local edit are identical to the earlier copy. These are controlled relative timing edits on original speech.

Each arm uses the production marked earlier-copy pipeline. The fallback keeps its existing visual selection and fixed 60 ms audio template. It supplies a second comparison only if production abstains for insufficient support. The result is useful when it has the correct sign within 80 ms and the entire reported audio excerpt lies inside the known shifted range. The frozen acceptance rule requires two newly localized actor groups, all incumbent useful locations and zero added control alerts.

| Delivery | Views | Production locations | Short template alone | Production with fallback |
| --- | ---: | ---: | ---: | ---: |
| Audio moved 240 ms earlier | 6 | 6 | 5 | 6 |
| Audio moved 240 ms later | 6 | 3 | 5 | 5 |
| Unchanged originals, alerts | 6 | 0 | 0 | 0 |
| Local gain only, alerts | 6 | 0 | 0 | 0 |

Every useful shift reproduced the exact signed offset. Independent readback re-decoded all 24 pairs, checked the original archive members and reconstructed the exact changed samples. It verified 82 picture, crop and PCM records from the reported evidence and 72 dispositions across the three arms.

The result applies to matching native GRID clips with a marked visual event and exact copied audio. More general timing changes, re-encoded speech and unmarked single-upload analysis require separate evaluation. This comparison carries zero weight in authorship confidence.

## Reproduce

Use the previously verified release qualification as `panel` and a new local directory as `run`. Use the production comparator revision pinned by the held-study commit. Keep source clips, speaker metadata and per-clip evidence local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze author score; do
  backend/.venv/bin/python -B -m scripts.grid_marked_release_holdout "$phase" "$panel" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.grid_marked_release_holdout review "$panel" "$run" \
  --public "$run/public-aggregate.json"
```
