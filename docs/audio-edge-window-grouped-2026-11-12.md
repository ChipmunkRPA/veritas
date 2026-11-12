# Prospective grouped calibration for short voice edits

The five overlapping two-second windows localized two of 12 new edits under strong competing speech. Full flagged one of those same edits and gave a four-second range. Neither grid flagged any of 14 genuine groups at the primary point, frozen with the same ordinary-group allowance.

## Held results

| Result | Full, two-group point | Combined, two-group point | Full, one-group point | Combined, one-group point |
| --- | ---: | ---: | ---: | ---: |
| Generated edits flagged of 12 | 1 | 2 | 0 | 1 |
| Flags covering at least half the known edit | 1 | 2 | 0 | 1 |
| Useful locations of 12 | 0 | 2 | 0 | 1 |
| Generated edits without a flag of 12 | 11 | 10 | 12 | 11 |
| Genuine false files of 38 | 0 | 0 | 0 | 0 |
| Genuine false groups of 14 | 0 | 0 | 0 | 0 |

Full flagged 2–6 seconds on a stock insert at 3.12–4.08 seconds under −5 dB competing speech. Combined narrowed the same finding to 3–5 seconds and found a clone insert at 5.04–5.68 seconds under −8 dB competing speech, with a 4–6 second review range. The two hits came from different meetings. The stricter point retained the stock location. Neither grid flagged either isolated room recording. All 50 files received at least one response in both grids. The Combined grid had a useful admitted candidate for all 12 edits and its strongest window met the location rule in seven cases before applying either point. The selected ranges support targeted listening. Ten misses at the primary point and the small genuine cohort limit the claim. [Anonymous holdout receipt](evaluation-panels/audio-edge-window-grouped-2026-11-12.json)

## Comparison and calibration

The earlier short-window trial found useful locations in retrospective rankings under strong competing speech. All four grids missed at their points set to the largest ordinary file response. This study compares Full and Combined at the same preselected genuine-group allowance, with new exact recordings, donors, and room excerpts. [Earlier trial and raw-score receipt](audio-edge-window-transfer-2026-11-12.md)

Full plans 0–4 and 2–6 seconds on continuous six-second audio. Combined checks 0–2, 1–3, 2–4, 3–5, and 4–6 seconds. Both use the installed commercially admitted voice checkpoint and its existing input preprocessing. The two grids use identical decoded PCM and speech activity. A candidate window requires detected speech on at least half of its duration. The strongest raw log-odds response nominates the range, with ties favoring the earlier start. [Full voice adapter study](audio-communication-serving-grid-2026-11-12.md) · [Checkpoint terms](commercial-model-audit.md)

For each grid, take one maximum score per genuine source group among 40 previously committed ordinary calibration clips across 16 groups. The primary point is the third-largest group maximum, with strict excess allowing two false calibration groups. The second-largest maximum supplies a frozen one-group sensitivity comparison. Both grids attained exactly two and one false calibration groups at their respective points. The points were written before any held inference and were unchanged for both noise levels. The native and public reviewers independently reconstruct the fit. These are experimental range-screening comparisons, with no customer probability. [Ordinary cohort and source terms](audio-remote-neighbor-replication-2026-11-12.md)

An interval has useful localization when it covers at least half the known edit and reaches 0.30 temporal intersection over union. The report also counts flags covering at least half even when their interval is too broad to meet the IoU criterion. The 0.64–0.96 second edits have a best possible IoU of 0.16–0.24 with any four-second Full range. Full can still flag an edit and cover it. Count a genuine group once if any of its controls flags.

## Held recordings

The held panel contains 50 six-second recordings from 14 groups. There are 12 generated inserts, 12 untouched hosts, 12 ordinary same-reader splices, 12 ordinary other-reader splices, and two isolated room recordings. The generated donors use four distinct voice profiles per family across DuplexGen PLN clone, ElevenLabs v4 expressive, and Kokoro stock. Declared speaker groups and exact audio inputs exclude both ordinary calibration and the earlier exploratory short-window panel. All eight expressive and stock profiles had unrelated earlier exposure, and their selected original recordings and decoded donor fragments were distinct. [Retained source and generation terms](audio-communication-replication-2026-11-12.md)

The matched inputs are identical outside their known inserts. Each of the two held AMI meetings has six edits and one isolated room capture. Both meetings use declared participants disjoint from calibration and the earlier exploratory panel. The new exact six-second excerpts exclude earlier bound capture intervals. Manual word times show at least two competing voices across 440–960 ms of each edit. Each generator family has two inserts at −5 dB and two at −8 dB whole-clip foreground-to-background ratio. Two inserts are near the start, two near the end, and eight cross an interior boundary of the nonoverlapping two-second grid. [AMI corpus terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Local evidence

In candidate mode, a completed primary voice check can trigger a separate short-window review on Full analysis of an exact six-second, 16 kHz, 16-bit mono PCM WAV or native FLAC with a verified continuous zero-start source clock. The lossless FLAC test reproduced the exact analyzed WAV samples across 90 calibration and held recordings. The review uses the primary two-group point. Its own analysis budget keeps a slow or unavailable review from discarding the completed primary response. The review supplies a flagged window, a no-flag result, or a skip reason. It carries zero weight in the overall assessment.

The flagged evidence stores the checked sample ranges, rational source timestamps, a WAV excerpt, and the full-recording waveform with the selected two seconds highlighted. The local worker reproduced both known flagged ranges and showed no flag for the matched genuine and isolated-room test files. Chrome's plain view loaded the waveform and the exact two-second excerpt from 3–5 seconds. The Standard PDF embeds the same waveform pixels as the JSON asset. The signed package verifier compared the selected WAV and regenerated the waveform from the original recording.

## Replay

The public receipt contains anonymous scalar scores, grouping, and known edit ranges. The reviewer checks the point ranks, misses, genuine-group errors and localization arithmetic. Native replay checks 50 complete PCM hashes and all 350 planned windows, verifies source independence and identical samples outside the known edit, and uses the saved responses.
The local holdout used one CPU checkpoint load and 349 admitted forwards.

```sh
python3 -B -S backend/scripts/review_audio_edge_window_grouped_transfer.py
```

With the pinned local assets and retained earlier inputs, use new directories to reconstruct the study.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_edge_window_grouped_transfer sources --held "$held"
backend/.venv/bin/python -B -m scripts.audio_edge_window_grouped_transfer freeze --run "$run" --held "$held"
backend/.venv/bin/python -B -m scripts.audio_edge_window_grouped_transfer score --run "$run" --held "$held"
backend/.venv/bin/python -B -m scripts.review_audio_edge_window_grouped_transfer --native "$run" "$held" --receipt "$run/replay.json"
```
