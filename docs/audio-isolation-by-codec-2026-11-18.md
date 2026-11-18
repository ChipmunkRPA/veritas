# Short speech edits with the same Opus encoding

Isolating the known foreground still improved detection when both the
foreground and mixture used the same Opus recipe. At both frozen points
the isolated Opus foreground located four of eight reachable edits clear
of their matched controls. The noisy Opus mixture reached two. This
comparison uses source stems available only for the benchmark. A second
experiment uses the submitted mixture alone and tests whether repeating
the encode can reject unstable regions.
[Earlier exact-foreground study](audio-exact-foreground-2026-11-18.md)
· [Study of audio detection and codec changes](https://arxiv.org/abs/2609.39651)

## Matched delivery comparison

The comparison reuses the 11 earlier groups with genuine matched utterances,
edited utterances from six generator families, and the corresponding room
recordings. It keeps the 750 ms windows, native sample positions, activity
admission and both earlier decision points. Two windows 500 ms apart must
both pass the same point. Only eight of 75 known edits can be located by the
fixed schedule. An edit counts when a proposal covers at least half the
marked samples and 0.30 of its temporal union. The ordinary and room
controls use the same coordinates. A clear location requires neither
control to alert at that location.
[Source and model terms](audio-exact-foreground-2026-11-18.md#fixed-comparison)

We scored lossless signed 16-bit PCM and mono 64 kbps constant-rate Opus at
both interference levels. The 20 ms Opus encoding recipe, decoder and sample
positions were identical. The replay verified the SHA-256 hashes of all 33
regenerated historical mixtures and room deliveries before scoring them and
their decoded samples. The 22 separately encoded foregrounds matched the
original presentation clock. Rounding the isolated foreground to PCM16
used the same samples in the lossless and encoded paths, without gain
normalization or clipping. The run made 218 single-CPU model calls.
[Opus carrier replay](audio-two-second-opus-transfer-2026-11-18.md)
· [FFmpeg Opus encoder options](https://www.ffmpeg.org/ffmpeg-all.html#libopus)

| Reading | Strict located · clear | Wider located · clear | Ordinary alerts, strict · wider | Wrong-time-only edited files, strict · wider |
| --- | ---: | ---: | ---: | ---: |
| Noisy PCM mixture | 2 · 2 | 2 · 2 | 2 · 2 | 1 · 1 |
| Noisy Opus mixture | 2 · 2 | 3 · 2 | 2 · 2 | 1 · 0 |
| Isolated PCM foreground | 3 · 3 | 6 · 6 | 0 · 0 | 2 · 1 |
| Isolated Opus foreground | 4 · 4 | 4 · 4 | 0 · 0 | 1 · 1 |

All lanes produced zero alerts on the 11 isolated room controls at both
points. Each noisy lane admitted all 16 scheduled pairs for each speech
role. Each isolated lane admitted 15 of 16 for each speech role. In Opus,
isolating the foreground added two clear locations across two groups and
retained both earlier clear locations at each point. At the wider point
the two additional PCM-only locations belonged to one group and shared
one reading. Encoding changed which isolated windows cleared a point,
while the matched Opus comparison still showed a gain from isolation.

## Repeat encoding of the submitted mixture

A separate branch rescored just the seven pairs that passed the earlier
wider point. It took the original decoded mixture through one additional
encode with the same settings and checked the same native crops. It
required both deliveries to pass at the same point. All 14 candidate
crops received a reading.

| Noisy Opus check | Strict located · clear | Wider located · clear | Ordinary alerts, strict · wider | Wrong-time-only edited files, strict · wider |
| --- | ---: | ---: | ---: | ---: |
| Original delivery | 2 · 2 | 3 · 2 | 2 · 2 | 1 · 0 |
| Require both encodes | 1 · 1 | 2 · 2 | 2 · 2 | 1 · 1 |

The repeat encode cleared no ordinary alerts. It lost one clear location
at the stricter point. It stays in evaluation. The isolated-foreground
comparison identifies a target for separation tests on real submissions.
Its counts apply to these selected crop positions and one Opus recipe.

## Offline recount

The numeric receipt includes group slots, decision bits and commitments to
the local plan and native readings. The public timing ledger supplies
the original Opus decisions and labeled intervals. The recount verifies
selection, matching, changed decisions and the model-call budget.
[Numeric receipt](evaluation-panels/audio-isolation-by-codec-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_isolation_codec \
  docs/evaluation-panels/audio-isolation-by-codec-2026-11-18.json.gz
```
