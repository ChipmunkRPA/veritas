# Original-audio scoring with paired splice boundaries

The independent boundary cue reduced false alarms in the eight held same-speaker clone cases at the frozen points. It localized the same one edit as the current voice check, and all three arms still localized one edit at matched false-group counts. The cue remains experimental. [Aggregate results](evaluation-panels/audio-original-pcm-boundary-transfer-2026-11-12.json)

## Sources and methods

Use the original unseparated mono mixtures from the source-qualified same-speaker study. Six selection speakers contribute 12 genuine controls for the operating points. Their six synthetic examples are scored separately after the points are locked. The eight confirmation speakers contribute eight 0.8-second same-speaker cloned inserts and 16 matched genuine splices. Their edits overlap at least two recorded competing voices for 650–800 ms. They are new inputs for the voice model and were already used for acoustic-only measurement. The transfer panel adds the exact earlier three expressive and three stock-synthesis episodes, with 12 matched ordinary splices and two isolated meeting-speech controls. The transfer scores have prior project exposure and do not qualify as a second prospective test. Its declared genuine host and ordinary-donor identities do not cross either new source role. [Same-speaker source qualification](audio-same-speaker-clone-sources-2026-11-12.md) · [Known room overlap](audio-same-speaker-enrollment-transfer-2026-11-12.md) · [Earlier room comparison](audio-spectra-long-room-holdout-2026-11-12.md)

The current check scores exact two-second intervals starting at 1, 3 and 5 seconds where the original permits and the speech gate admits the window. The exposed transfer uses the already saved native current responses after checking their input hashes. The separate cue compares adjacent 160 ms distributions of power in 24 mel-frequency bands at both ends of a proposed 0.8-second span. The endpoint comparison excludes the nearest 20 ms on each side and uses Jensen–Shannon divergence. Candidate spans begin every 80 ms. The weaker endpoint defines the cue. The combination averages standardized cue and current responses only where a native current window contains the full candidate. All comparisons use the original submitted PCM. The reference samples and known edit stems do not enter inference. The current checkpoint, its encoder and the speech gate retain their existing publisher MIT declarations. [Commercial model audit](commercial-model-audit.md) · [Matched benign-edit research](https://arxiv.org/abs/2610.04479) · [Seamless edit research](https://arxiv.org/abs/2501.03805)

Each arm flags a clip only when its score strictly exceeds the largest eligible genuine selection score. The combination uses median and interquartile scaling estimated on candidate windows from those same genuine clips. A useful range covers at least half of the edit and reaches 0.30 temporal intersection over union. The retrospective 0–2 false-group counts measure ranking only. An alarm on either matched genuine splice counts against its source group. The candidate duration matches this panel's controlled geometry. No evidence about seamless edits or other durations follows from this design.

Before speech admission or any response, the native interval geometry can satisfy the location rule on four of the eight held same-speaker edits for the current windows, all eight for the shorter cue, and six for the supported combination. The corresponding counts for the six exposed alternative edits are three, six, and four. A shorter output interval and wider coverage of the clip can change localization even before score quality enters the comparison.

## Held results

| Panel and frozen arm | AI flags | Useful ranges | False genuine groups |
| --- | ---: | ---: | ---: |
| Same-speaker clone, current | 4 of 8 | 1 | 5 of 8 |
| Same-speaker clone, boundary | 1 of 8 | 1 | 0 of 8 |
| Same-speaker clone, combined | 1 of 8 | 1 | 0 of 8 |
| Earlier expressive and stock, current | 1 of 6 | 1 | 1 of 8 |
| Earlier expressive and stock, boundary | 2 of 6 | 2 | 0 of 8 |
| Earlier expressive and stock, combined | 2 of 6 | 1 | 0 of 8 |

In the held same-speaker panel, all three arms localized the same edit. The recorded edit spans 3.84–4.64 seconds. The current check nominated 3.00–5.00 seconds, and the cue and combination each nominated 3.92–4.72 seconds. Each arm localized one held clone at zero, one, and two false genuine groups. Before applying the points, the strongest current window located four of the eight clone edits and the boundary and combined windows each located two. Five false genuine groups under the current fixed point account for eight flagged controls, four same-speaker and four other-speaker splices.

The older transfer's two useful boundary findings both came from expressive synthesis in one room. At zero, one and two retrospective false groups, the boundary localized two earlier edits and the current check localized one, two, and two. The boundary flagged neither isolated-room control. The earlier transfer gives a lead for a new room and generator test, and the held same-speaker results give no measured gain in the number of correctly located edits.

## Reproduction

The exact current detector made 21 point and 48 held and diagnostic CPU forwards. It reused identical native windows within each stage only when their PCM hashes matched. The exposed transfer used its separately committed original current detector responses. The independent review rebuilt the 138 native window hashes, the exact positions of 4,070 paired boundary ranges, the cutoffs, and both sets of false-group counts. The native replay also recomputed every boundary value from the source PCM. Neither audit reran neural inference. Raw recordings and per-file scores remain local.

```sh
base=backend/data/independent-evaluation
run="$base/audio-original-pcm-boundary-2026-11-12"
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_original_pcm_boundary_transfer audit "$run"
python3 -B -S backend/scripts/review_audio_original_pcm_boundary_transfer.py "$run" \
  "$base/audio-same-speaker-enrollment-2026-11-12" \
  "$base/audio-spectra-long-room-final-2026-11-12"
```

To reconstruct a separate receipt, choose a new local run directory and execute `freeze`, `point`, `held`, and `report` in that order before both audits. [Runner](../backend/scripts/audio_original_pcm_boundary_transfer.py) · [Independent reviewer](../backend/scripts/review_audio_original_pcm_boundary_transfer.py)
