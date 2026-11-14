# Can changing the speech window recover short edits under noise?

The current focused review looks for agreement between overlapping
three-second voice responses. We tested whether moving the search grid by
750 ms or using 2.5-second windows could catch edits that crossed the current
window boundaries. Both comparisons use the existing speech-activity check,
voice classifier, and calibration on the original mixture. They add no
external weights and produce no new overall confidence estimate. Boundary
localization and the classification of the resulting content are separate
problems. [Split and Conquer Partial Deepfake Speech](https://arxiv.org/abs/2604.02913)
· [Temporal Anchors and Editing Sensitivity](https://arxiv.org/abs/2610.04479)
· [Installed asset terms](commercial-model-audit.md)

## Matched comparison

The staggered three-second arm still needs two AI-leaning windows whose
starts are 1.5 s apart. The 2.5-second arm requires an existing AI-leaning
three-second window within 750 ms by midpoint. Both arms present the
overlap and keep previous evidence. The fixed rule for a fresh confirmation
required at least three added locations under noise across two generator
families, at most one new genuine group and no gain-only or isolated
background flags.

The 98 exact deliveries reproduce the earlier comparison. Eighteen partially
edited recordings from six generators and 12 genuine recordings each have a
gain-only, rain and train delivery. Eight background-only excerpts supplement
those matched views. Rain and train were mixed about 5 dB above the foreground
RMS. All these foregrounds and backgrounds have previous project exposure.
A location must cover at least half a labeled edit and reach temporal IoU
of 0.30. [Original inputs, edits and licenses](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)

| Outcome | Existing review | With staggered 3 s windows | With 2.5 s windows |
| --- | ---: | ---: | ---: |
| Gain-only edits located | 13/18 | 15/18 | 14/18 |
| Rain-masked edits located | 3/18 | 4/18 | 3/18 |
| Train-masked edits located | 0/18 | 0/18 | 0/18 |
| Genuine deliveries flagged | 0/36 | 0/36 | 0/36 |
| Isolated background flags | 0/8 | 0/8 | 0/8 |

The staggered arm added one useful rain location from XTTS v2. It also
proposed an extra interval on that clip. The 2.5-second arm added a
misplaced flag on a rain-edited delivery. Both arms fell short of the
frozen noise requirement. The gain-only improvement can be tested on a
separate clean panel. Neither experimental route entered serving.

On the train-masked edits, ten files could have produced a useful
staggered overlap if every planned window had passed. Zero still had that
geometry after the speech gate. The 2.5-second arm had 13 geometrically
reachable files, three after speech admission and zero after scoring. This
isolates two distinct losses on this panel. Repositioning the windows did
not overcome either one. [Window dispositions and recount](evaluation-panels/audio-staggered-voice-2026-11-14.json)

## Coverage in the app

The long speech review now groups adjacent source samples by whether a
covering window gave a directional response, only inconclusive responses,
or no score because detected speech was too short. It also identifies
parts of the source outside the schedule. Overlap counts once. The
table describes which windows reached a response, while the separate
finding still marks the intersection supporting a specific lead.

The app, API and PDF show the same coverage, and the signed JSON stores
the exact source sample intervals. In six simulated prepaid runs, no job
started before payment and overall decisions agreed with the 48 kHz trial
off and on. The edited rain example had 4.500 s of directional window
coverage, 2.715 s of inconclusive coverage and a separate **3.000–4.500 s**
corroborated finding. The rain-only control was below the speech minimum
for all eight seconds. The signed replay verified the original PCM and
speech counts for the edited sample. [Local app receipt](evaluation-panels/audio-staggered-voice-app-2026-11-14.json)

## Replay

Use the original study's reconstruction to prepare the same `roster.json`
and PCM directories locally. Published measurements contain durations and
window directions without speech samples or model weights.

```sh
export PYTHONPATH=backend VERITAS_AUDIO_DETECTOR_MODE=candidate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipt=docs/evaluation-panels/audio-staggered-voice-2026-11-14.json
python3 -B -S -m scripts.audio_staggered_voice_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.audio_staggered_voice_transfer run "$run" --native "$parent"
cmp "$receipt" "$run/receipt.json"
```
