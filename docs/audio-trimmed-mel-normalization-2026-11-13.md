# Robust Mel normalization for short edits under competing speech

The source-isolation diagnostic found useful locations on five of 12 known
edits when the model saw the exact clean foreground, and none when it saw the
blind separator outputs. This experiment keeps the original waveform and
asks whether dampening the influence of the strongest and weakest log-Mel
frames on normalization improves localization. It uses the same model and
does not reconstruct, separate or otherwise filter the recording.
[Source-isolation diagnostic](audio-source-isolation-diagnostic-2026-11-13.md)
· [Quantile feature normalization in speech recognition](https://www.isca-archive.org/eurospeech_2001/hilger01_eurospeech.pdf)
· [Audio deepfake front-end comparison](https://www.isca-archive.org/interspeech_2022/muller22_interspeech.pdf)

## Fixed comparison

Use the original six-second mono mixture on the 16 kHz source clock. Keep
the standard waveform preprocessing, the 80 Mel bands, the speech gate, the
four-second contexts starting at 0 and 2 seconds, and the trained encoder
and classifier. Estimate a second mean and sample variance from a copy of
each log-energy band clipped to that context's 10th and 90th percentiles.
Blend those two estimates equally with their conventional counterparts.
Normalize the original feature values using the blended moments. The
normalized features retain their original temporal order and do not have
their tails clipped. The operation changes the coordinate scale the model
sees, and can still weaken a cue.

Read the unchanged affine classifier on contextual frames, take the median
in each one-second interval on a half-second grid, and use the lower median
where both four-second contexts cover that interval. Each eligible interval
must pass the original speech gate. The comparator uses the same geometry
with conventional normalization. The older fixed two-second check is a
separate comparator. The installed voice and speech gate assets are the
same ones already evaluated. No new weights or third-party method code
are needed. [Asset terms](commercial-model-audit.md)

The 40 development deliveries contain 28 ordinary controls. Seal the new
point at the highest admitted ordinary development response. The 52 held
deliveries contain 12 known synthetic insertions across three generator
families, 36 matched ordinary clips and four isolated room controls. Held
speakers and rooms are separate from development. The exact waveforms
were scored before for other methods. The voices do not establish
pretraining separation. Scoring receives no reference track, edit timing
or source labels. [Source qualification](audio-remote-neighbor-replication-2026-11-12.md)

A useful location covers half of the known edit with at least 0.30 temporal
intersection over union. Count ordinary alarms once per matched source
group. To qualify for fresh evaluation, the new route must keep the old
useful result, add two across two generator families, add no more than one
ordinary group, and flag no isolated room. Descriptive equal-alarm sweeps
use held results and do not select a serving point.

## Held outcomes

| Method | Useful edits / 12 | Ordinary flags / 40 | Ordinary groups flagged / 16 | Useful strongest range before the point / 12 |
| --- | ---: | ---: | ---: | ---: |
| Current two-second check | 1 | 7 | 3 | 5 |
| Conventional normalization, dense ranges | 0 | 0 | 0 | 3 |
| Robust normalization, dense ranges | 0 | 0 | 0 | 3 |

All 52 deliveries were assessed by all three methods. The candidate's
development point was **7.045034** on the raw logit scale. It produced
no held flag. Both dense readouts had an admitted geometrically useful
range on every edit. At descriptive budgets of zero, one, two and three
ordinary groups, every method located respectively zero, zero, one and
one useful edit. Both dense views scored nine of 12 edits higher than both
matched ordinary splices when restricted to jointly admitted useful ranges
selected using benchmark timing. The fixed two-second check did so on five.
The known edit times are unavailable during a normal analysis. The dense
ranking result therefore measures an available local response alongside
an unresolved interval-selection problem. The change to the normalization
did not improve either result. Keep the current app check and confidence.
[Anonymous responses and decisions](evaluation-panels/audio-trimmed-mel-normalization-2026-11-13.json)

## Replay

The portable reader recalculates the selected point and source-interval
outcomes from the published scalar scores and the pinned parent receipt.
Use Python 3.11 or later.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_trimmed_mel_normalization review \
  docs/evaluation-panels/audio-trimmed-mel-normalization-2026-11-13.json
```

For native replay, retain the source folders used by the isolation diagnostic
and the existing checkpoint. Choose a new run directory. The runner verifies
the mixture hashes and earlier source commitments. A conventional forward
rechecks archived interval scores at the start of each bank. The run made
66 development and 86 held candidate forwards. The published receipt
contains no waveforms, identities or model features.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze development assessment; do
  backend/.venv/bin/python -B -m scripts.audio_trimmed_mel_normalization "$phase" "$run" || exit 1
done
```
