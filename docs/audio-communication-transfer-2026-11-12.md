# Short voice edits after simulated call processing

Requiring a flag on the same audio window before and after a fixed processing chain retained the one useful original location and reduced false genuine groups from three to two. The chain alone found two useful locations and flagged six genuine groups. These are new processed deliveries of an already scored source panel. The paired check needs confirmation on different source groups before serving use. [Aggregate and replay](evaluation-panels/audio-communication-transfer-2026-11-12.json)

## Matched comparison

Use all 52 previously committed six-second mono recordings with competing room speech. Twelve source groups each supply one generated insert, one genuine same-reader splice, one genuine other-reader splice and an unedited host. Four additional examples contain room sound alone. The generated inserts cover three synthesis families. Their source ranges and the retained native voice scores are bound by the earlier replication. [Sources, rights and prior score ledger](audio-remote-neighbor-replication-2026-11-12.md)

Before new inference, freeze a 300–3,400 Hz bandpass, a 3:1 compressor above 0.1 amplitude with 10 ms attack and 80 ms release, and constant-rate 24 kb/s Opus encoding using the voice setting. Retain each compressed file and re-decode through the app. All 52 decoded files retain exactly 96,000 samples starting at tick zero. This is a controlled transformation of the same recordings. Call platforms and capture hardware can use other chains. [RADAR media transformation protocol](https://arxiv.org/abs/2605.09568) · [Acoustic front-end study](https://arxiv.org/abs/2607.17761)

Compare the existing licensed voice checkpoint on the original and processed 1–3 s and 3–5 s windows. Apply the same speech gate on each view independently and retain the strongest admitted window. The paired arm takes the lower response at each shared admitted interval. Every arm uses the original fixed point. A useful interval reaches 0.30 temporal intersection over union and covers at least half the known insert. A false group is any group with a flag on at least one genuine delivery. [Checkpoint and source terms](commercial-model-audit.md)

## Local results

| 12 generated and 40 genuine views | Original | Processed | Both views |
| --- | ---: | ---: | ---: |
| Generated flags at frozen point | 3 | 5 | 2 |
| Useful generated locations at frozen point | 1 | 2 | 1 |
| False genuine groups at frozen point | 3 | 6 | 2 |
| Useful locations at 0 / 1 / 2 equal false groups | 0 / 0 / 1 | 1 / 1 / 1 | 0 / 0 / 1 |
| Unassessed files | 0 | 0 | 0 |

The paired check preserved the same useful interval and removed one genuine group without adding a false group. It also withheld one AI flag whose location missed the known edit. All four room-only controls stayed below the point in all arms. Both views score the same underlying samples and share one classifier. Agreement gives a reproducible stability observation and carries no AI-confidence weight. The broad failure to locate short inserts under strong competing speech persists.

## Replay

Use the retained source panel above and the pinned local voice checkpoint. The first step freezes delivered PCM and the original cutoff before scoring. The audit decodes all 52 stored Opus deliveries again. The separate standard-library reviewer reconstructs the three fixed decisions and verifies all 208 native input-window commitments and the exact retained location. Neither replay reruns classifier inference.

```sh
base=backend/data/independent-evaluation
run="$base/audio-communication-transfer-2026-11-12"
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in prepare score report audit; do
  backend/.venv/bin/python -B -m scripts.audio_communication_transfer "$phase" "$run" || exit 1
done
python3 -B -S backend/scripts/review_audio_communication_transfer.py \
  "$run" "$base/audio-remote-neighbor-replication-2026-11-12"
```
