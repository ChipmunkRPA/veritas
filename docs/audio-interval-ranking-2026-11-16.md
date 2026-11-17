# Directly ranking short speech intervals

The candidate scores each complete 250–1000 ms interval. It combines
the mean, spread and frame-to-frame changes of time-aligned speech features.
The relative arm additionally compares the interval with its own nearby
speech. All versions use the original 16 kHz clock, a 50 ms search step and
the existing speech-activity check. The measurements use the existing
commercially permitted encoder. No new pretrained weights were added.
[Encoder and source terms](audio-wavlm-v13-protocol.md)

Recent partial-speech detection research studies interval content separately
from changes at the boundaries. The local fit uses that distinction and
ranks complete intervals internally. The first fit compares each annotated
interval with the same time in an original, an ordinary resampling and an
isolated background. The second fit also includes the highest-scoring wrong
intervals in the development recordings. Each of the three fits uses 10,332
matched comparisons from 64 prompts. Separately, 24 genuine calibration
prompts supply 264 quiet, reprocessed and noisy control deliveries.
[Window-level speech localization](https://arxiv.org/html/2604.02913)
· [Boundary and within-segment research](https://arxiv.org/html/2609.25822v1)

## Results on the same audio clock

The first comparison uses 312 deliveries from the earlier six-second study.
It compares six edit generators and keeps all test speakers outside the
interval fit and calibration. The candidate in the table uses both interval
measurements and their contrast with neighboring speech. It can add a range
when the earlier longer-file scan and its six-second fallback propose none.
Correct ranges cover at least half of a labeled edit and reach 0.30 temporal
intersection over union. [Earlier panel and custody](audio-six-ten-context-2026-11-16.md)

| Condition | Correct locations, current → combined | Untouched flagged, current → combined | Ordinary processing flagged, current → combined | Isolated background flags |
| --- | ---: | ---: | ---: | ---: |
| Clean | 35 → 35 of 73 | 1 → 1 of 24 | 1 → 1 of 24 | — |
| Competing voice 5 dB louder | 21 → 21 of 73 | 7 → 7 of 24 | 7 → 7 of 24 | 0 of 48 |
| Competing voice 10 dB louder | 11 → 12 of 73 | 5 → 5 of 24 | 5 → 5 of 24 | 0 of 48 |

The matched diagnostic uses 112 separate deliveries, including untouched
versions of the exact same utterances. These recordings share source
groups with the 312-delivery comparison. The comparator here is the
previous six-second check alone. On clean speech the wider candidate
adds one of 42 edited locations and one misplaced range. Under a voice
5 dB louder it adds no correct locations, one misplaced range and one
untouched alert. Under a voice 10 dB louder it adds one misplaced range
and no correct locations. The stricter candidate adds no locations on
the matched panel. The candidate remains in local evaluation.
[312-delivery receipt](evaluation-panels/audio-interval-ranking-held-2026-11-16.json.gz)
· [Matched originals and edits receipt](evaluation-panels/audio-interval-ranking-matched-2026-11-16.json.gz)

## Replay

The receipts contain the fitted parameters, numeric readings, test
positions and input hashes. Raw recordings and intermediate speech
features stay local. The standard-library check recalculates the
operating points and event counts. A second execution of the native
matched comparison reproduced its receipt byte for byte.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
python3 -B -S -m scripts.audio_interval_ranking_review "$panels/audio-interval-ranking-held-2026-11-16.json.gz"
python3 -B -S -m scripts.audio_interval_ranking_review "$panels/audio-interval-ranking-matched-2026-11-16.json.gz"
```

Restore the original fitted inputs, acoustic control manifests and pinned
assets following the [six-second replay](audio-six-ten-context-2026-11-16.md#replay).
Use the same retained fit files, prospective feature cache and matched
source files. Each reader verifies the digests before using cached inputs.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m scripts.audio_interval_ranking_fit fit "$bank" "$out/fit.json.gz"
"$runner" -B -m scripts.audio_interval_ranking_fit compare "$bank" "$out/fit.json.gz" "$out/held.json.gz"
"$runner" -B -m scripts.audio_interval_ranking_matched "$bank" "$matched" "$cache" "$out/fit.json.gz" "$out/matched.json.gz"
```
