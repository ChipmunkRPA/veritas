# Adjacent speech-frame changes under loud background noise

Adding this cue to the focused speech check found one extra location under
rain and flagged two genuine rain recordings. The check stays experimental.

| Held deliveries | Focused check | With the new cue |
| --- | ---: | ---: |
| Gain-only edits localized | 13/18 | 13/18 |
| Rain edits localized | 3/18 | 4/18 |
| Train edits localized | 0/18 | 0/18 |
| Genuine gain-only recordings flagged | 0/12 | 0/12 |
| Genuine rain recordings flagged | 0/12 | 2/12 |
| Genuine train recordings flagged | 0/12 | 0/12 |
| Isolated background excerpts flagged | 0/8 | 0/8 |

The extra rain result covered 5,328 samples of an 8,800-sample edit from
one XTTS-v2 source. The selected range was 3.000–3.800 seconds. The two
new false alarms marked 1.600 seconds across genuine recordings. The new
range also marked 0.467 seconds of genuine speech inside the edited file,
for 2.067 seconds of additional incorrect highlighted time overall.
Before applying the cutoff, the best range matched a useful edit in
2 gain-only, 6 rain, and 3 train recordings.

## Fixed comparison

The published TRACE study measures changes between adjacent normalized
speech embeddings and summarizes them for whole-recording classification.
This independent test uses the existing, pinned WavLM Base+ weights and
one local statistic. It uses layer 10, the paper's best reported single
layer for the related Base checkpoint. One forward on the exact mono
16 kHz delivery supplies the embeddings. Each candidate range spans
0.8 seconds on a 0.2-second grid and requires at least half its duration
to pass the pinned speech activity check. The feature is the RMS of
distances between adjacent unit-normalized embeddings, using pairs whose
midpoint falls inside the candidate range. Choose the earliest best
range and require it to exceed the highest admitted genuine selection
score. Preserve all focused ranges. [TRACE paper](https://arxiv.org/html/2604.01083v1)
· [Installed checkpoint terms](audio-wavlm-v13-protocol.md)
· [Speech activity terms](commercial-model-audit.md)

The operating point came from 22 admitted deliveries among 25 genuine
speech and isolated-background selection controls. The unchanged held
panel has 98 deliveries from 30 speech groups and two background
recordings. The new cue assessed 85, including all 54 edits. Five genuine
train mixtures and all eight isolated backgrounds had no admitted range.
The held recordings already had focused and frequency-view results.
Speaker exposure and encoder-training overlap remain possible. A useful
range covers at least half of one exact labeled edit and reaches 0.30
temporal intersection over union. The study uses its own implementation
with installed weights and retains all media locally. An embedding change
can reflect a natural speech transition or a change in the recording.
[Exact sources, licenses and prior tests](audio-subband-partial-transfer-2026-11-12.md)

## Replay

Two CPU runs produced byte-identical receipts. The offline reviewer
recomputes the speech-admitted range schedule, the operating point and
all decisions from saved scalar measurements. Native replay also checks
the 123 delivery hashes. Only the separate inference run recomputes
speech activity and the encoder responses. The public receipt contains
sample ranges and scalar measurements without waveforms or embeddings.
[Receipt](evaluation-panels/audio-adjacent-frame-transfer-2026-11-12.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_adjacent_frame_transfer review \
  docs/evaluation-panels/audio-adjacent-frame-transfer-2026-11-12.json
```

Use the exact staggered-study selection and focused-study held directories
from the source steps above. Use a new directory for `run`.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
backend/.venv/bin/python -B -m scripts.audio_adjacent_frame_transfer run "$run" \
  --selection "$selection" --held "$held"
backend/.venv/bin/python -B -m scripts.audio_adjacent_frame_transfer review "$run/receipt.json" \
  --selection "$selection" --held "$held"
```
