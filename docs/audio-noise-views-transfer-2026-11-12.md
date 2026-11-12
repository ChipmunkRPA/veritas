# Separate decision rules for enhanced speech under loud backgrounds

An earlier traffic study recovered 44 synthetic flags after speech
enhancement and added 39 genuine false flags. The same decision rule was
being applied before and after signal processing. This comparison gives
the enhanced response its own operating point and tests whether the
original waveform supplies complementary information.
[Earlier enhancement test](audio-traffic-enhancement-transfer-2026-11-11.md) ·
[Speech enhancement and detector transfer](https://arxiv.org/abs/2603.14767) ·
[Noise-aware speech detection evaluation](https://arxiv.org/abs/2512.13744)

## Frozen design

The fit and point-setting sets use the 40 exact English source recordings
from the separate RADAR development split. Within each source class, use
the first 12 hash-ordered recordings to fit and the other eight to set
operating points. Each exact three-second center excerpt contributes a
gain-only control and two mixtures. Mix at −5 dB or −10 dB relative to
the recorded background according to the fixed within-class rank.
Use a single peak-safe gain for all three versions of each recording.
Each noise tape contributes its first four previously screened excerpts.
Speaker and generator identities are unavailable, so separation is by
publisher split and exact input. [RADAR source verification](audio-radar-transfer-2026-11-12.md)

The fit uses the street recording from Berlin and a kitchen tidy-up
recording. The operating points use a different traffic recording and
a separate kitchen ambience. Excerpts retain the frozen screening
status. The earlier automated screening set aside clips with detected
speech or music, and human listening review remains pending.
[Berlin traffic, CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Ambient_sound_street_traffic_Berlin_2026-05-17.oga) ·
[Kitchen tidy-up, public domain](https://commons.wikimedia.org/wiki/File:Kitchen_tidy_up.ogg) ·
[Night traffic, CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Cars_Passing_by_at_Night.webm) ·
[Kitchen ambience, CC0](https://commons.wikimedia.org/wiki/File:Ambient_Kitchen_Sounds.wav) ·
[Source preparation and credits](audio-commons-background-v234-method.md) ·
[Earlier screening](audio-commons-slice-screen-v247-result.md)

The pinned, MIT-licensed CPU enhancement model processes each exact
input while preserving its sample count. Score the original and
enhanced view with the existing voice model. The original view can
supply a raw response even when the ordinary activity check abstains.
Neither candidate can report evidence unless the enhanced waveform
passes speech admission. The first candidate uses its enhanced
response alone. The second fits a regularized linear score from the
original and enhanced logits and the change in signal energy. The
fit never reads held responses. For each candidate, seal the point at
the highest eligible genuine or isolated-background response in the
separate calibration role. These scores are experimental decision
variables and have no calibrated probability interpretation.
[Pinned enhancement and terms](audio-traffic-enhancement-transfer-2026-11-11.md#frozen-comparison) ·
[Voice assets and terms](commercial-model-audit.md)

The held comparison reproduces the exact 96 gain-only, −5 dB traffic and
−5 dB café views on the 32 short RADAR evaluation inputs. Eight isolated
background excerpts are added as controls. These foregrounds and
recordings are outside both development roles. The three conditions
repeat each source voice, so count unique foreground groups as well
as deliveries. The existing detector's results on these inputs were
already known from the RADAR baseline study. Both new heads were sealed
before their held inference.

A candidate can add an AI flag only when the incumbent is inconclusive
or unassessed. The follow-up criterion requires at least five additional
synthetic groups across both held noise scenes, up to three additional
genuine groups, and no added flag on an unoverlaid genuine recording or
an isolated background. The public replay also computes a descriptive
tradeoff at zero to three added genuine groups from the held scores.
This sweep does not set a release point.

## Held results

The enhancement made more synthetic foregrounds assessable. Traffic
coverage rose from 8/16 to 16/16, and café coverage rose from 5/16 to
13/16. That improvement in speech admission did not translate into
reliable synthetic-voice flags.

| Held view | Incumbent AI flags | With enhanced-only score | With joint score |
| --- | ---: | ---: | ---: |
| Traffic, synthetic foregrounds | 3/16 | 4/16 | 12/16 |
| Traffic, genuine foregrounds | 1/16 | 1/16 | 11/16 |
| Café, synthetic foregrounds | 0/16 | 0/16 | 5/16 |
| Café, genuine foregrounds | 0/16 | 0/16 | 5/16 |
| Gain-only, synthetic foregrounds | 11/16 | 11/16 | 12/16 |
| Gain-only, genuine foregrounds | 0/16 | 0/16 | 1/16 |
| Isolated background, candidate only | — | 0/8 | 1/8 |

The enhanced-only rule added one synthetic group with no added genuine
flag. The joint rule added 11 synthetic groups across both noisy
scenes, along with 12 genuine groups. No candidate passed. Even a
retrospective sweep of the held scores found at most three additional
synthetic groups within the allowed false-alarm budget. Both candidates
stay experimental.

An independent reader checked the exact bytes of all 240 prepared PCM
inputs. It recounted the 104 held decisions for each head and reproduced
the diagnostic tradeoffs without importing either model. The public
receipt contains source commitments and score records. Media stays local.
[Decisions and replay receipt](evaluation-panels/audio-noise-views-transfer-2026-11-12.json)

## Replay

Use the locally reproduced RADAR sources, including the `panel` and
`confirmation` directories and both original ambient recordings. The
development backgrounds are the normalized `slot-XX.pcm.wav` files
from the published Commons preparation. The phase runner verifies the
pinned enhancer bytes and MIT notice before running. ONNX Runtime and
the existing backend audio dependencies are required for inference.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
backend/.venv/bin/python -B -m scripts.audio_noise_views_transfer prepare "$run" \
  --assets "$enhancer_assets" --radar-source "$radar_sources" --commons-source "$commons_wavs"
backend/.venv/bin/python -B -m scripts.audio_noise_views_transfer development "$run" --assets "$enhancer_assets"
backend/.venv/bin/python -B -m scripts.audio_noise_views_transfer fit "$run"
backend/.venv/bin/python -B -m scripts.audio_noise_views_transfer held "$run" --assets "$enhancer_assets"
backend/.venv/bin/python -B -m scripts.audio_noise_views_transfer report "$run"
```

```sh
python3 -I -S backend/scripts/review_audio_noise_views_transfer.py review \
  docs/evaluation-panels/audio-noise-views-transfer-2026-11-12.json
```
