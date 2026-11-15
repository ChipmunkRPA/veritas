# Measuring short voice edits with an explicit boundary cue

The current technical question is whether a short change in speech
features helps locate synthetic words under a louder conversation.
Recent partial-audio research learns from changes between adjacent
features separately from the frame prediction. A separate study found
that ordinary speech restoration can look suspicious to classifiers.
We independently implemented a small boundary comparison and tested
it against genuine splices, resampling and isolated room backgrounds.
[Boundary learning study](https://arxiv.org/html/2609.25822v1#S2.SS1)
· [Ordinary restoration study](https://arxiv.org/abs/2603.14033)

Each model receives a single six-second recording. The content output
reads the original learned representation. The boundary output reads
the absolute change between its learned features 40 ms before and 40 ms
after each point, extending the edge values at the start and end of the
clip. Both versions have 3,626 fitted parameters and use the same frozen
speech features, initialization, fitting losses, and 0.50, 0.75 and
1.00 second proposals. Each model has its own strict and wider points,
set using genuine calibration speakers before the held evaluation.
The experiment uses the existing commercial-use-reviewed encoder and no
research implementation or new external weights.
[Prior architecture and input terms](audio-short-edit-speech-layers-2026-11-14.md)

## Primary panel

Each training direction evaluated the other generator. The 48 synthetic
edits, 144 ordinary recordings and 48 isolated-background tests reuse
the prior ICSI mixtures at −5 dB. A useful interval covers at least half
of the true edit and reaches 0.30 temporal intersection over union.
The reviewer separately counts a useful interval with a flag on the
same edit clock in any matched genuine or background control.

| Research check | Useful locations /48 | Useful without a matched alert | Ordinary flags /144 | Isolated-background flags /48 |
| --- | ---: | ---: | ---: | ---: |
| Prior boundary output, strict | 1 | 0 | 2 | 2 |
| Explicit boundary cue, strict | 1 | 1 | 0 | 0 |
| Prior boundary output, wider | 2 | 1 | 20 | 4 |
| Explicit boundary cue, wider | 3 | 3 | 20 | 3 |

Ordinary flags shifted between the two fits. The fit tested on Chatterbox
increased from 16 to 20 ordinary flags at the wider point. The fit tested
on MeloTTS reduced ordinary flags from four to zero. All three useful
wider locations are additions relative to the current short review on
this panel. Using both checks would also add 11 ordinary flags and three
isolated-background flags. Before cutoff, the explicit cue put 16 of 48
edits above all three of their matched ordinary recordings, down from 18.

## Transfer to separate recordings and generators

The fixed heads next read 172 retained deliveries, with PCM commitments
and source exclusions replayed against the previous manifests. The
48-file room bank uses four different meetings and three additional
generator families, all at −5 dB. The 124-file stock bank uses two
additional generator families and dry, 0 dB and −5 dB conditions. These
exact recordings had earlier project exposure and were outside both
fitting and point selection for this comparison.
[Room source terms](audio-remote-neighbor-replication-2026-11-12.md)
· [Stock source terms](audio-stock-voice-transfer-2026-11-13.md)

On the room bank, neither strict fit found a useful location. The wider
MeloTTS-trained fit found **2/11** with **9/33** ordinary flags. The
same prior fit found **1/11** with **3/33** ordinary flags. The current
short review found **0/11** with **5/33** ordinary flags. None of these
routes flagged an isolated room. The two boundary locations were:

- **Kokoro, 1.500–2.000 s** within the known **1.040–2.000 s** edit.
  Its matched genuine recordings stayed below the candidate's point.
  The prior research fit and the current review did not flag a useful location.
- **ElevenLabs, 3.700–4.200 s** within the known **3.440–4.240 s** edit.
  A matched genuine splice also flagged at exactly **3.700–4.200 s**.
  The prior research fit already located this edit.

Neither boundary fit found a useful Piper or KittenTTS location among
the 32 noisy stock edits. The current short review located **8/32**, with
**11/48** ordinary flags. Each dry location found by the boundary fits
had a matched ordinary alert at the edit clock. Both fits assessed all
172 deliveries. The isolated-background results, primary gains and the
single new Kokoro location are reproducible. The stock misses and
room-bank false alarms leave the cue in evaluation.

## Replay

The numeric receipts contain the response values, sample intervals and
source hashes. Recompute the points and all measured outcomes with
standard-library Python. Speech, representations and weights stay local.
A second CPU run reproduced both fitted weight files and the full
transfer receipt byte for byte. The earlier architecture also
reproduced its published weight commitments and responses.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
module=scripts.audio_temporal_boundary_transfer
candidate="$panels/audio-temporal-boundary-fit-2026-11-15.json"
transfer="$panels/audio-temporal-boundary-transfer-2026-11-15.json"
python3 -B -S -m "$module" review "$panels/audio-short-edit-detector-training-2026-11-14.json" \
  "$panels/audio-short-edit-speech-layers-2026-11-14.json" "$candidate"
python3 -B -S -m "$module" review-transfer "$transfer" "$candidate"
```

Native replay uses the source bank and feature cache from the
[frozen feature study](audio-short-edit-speech-layers-2026-11-14.md#independent-replay).
Rebuild the room roster and stock signals as described in the
[previous transfer replay](audio-competing-speech-ranking-2026-11-15.md#independent-replay).

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" fit "$sources" "$cache" "$speech_asset" "$run/fit"
OMP_NUM_THREADS=3 "$runner" -B -m "$module" transfer "$sources" "$rooms" "$stock" \
  "$encoder" "$speech_asset" "$run/fit" "$run/transfer.json"
```
