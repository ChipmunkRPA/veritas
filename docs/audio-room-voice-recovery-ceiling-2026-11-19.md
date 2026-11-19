# Exact room-voice recovery ceiling

The existing brief speech check found no additional useful edit ranges when
given the exact room-rendered voice instead of the two-voice mixture. The
existing two-second check retained one useful stock-voice range in each
view. All twelve edited voice clocks had adequate sampling and speech
admission. This bounded result identifies a classifier or calibration
limitation that remains after the other voice is removed.

## Frozen comparison

Select the earliest retained scene for each of six held reader pairs from
the [two-voice benchmark](audio-two-voice-components-2026-11-15.md). Two
pairs use PhonemeDF and four use MLS-derived stock voices. These sources
already appeared in earlier project evaluations. The twelve source voice
instances and one measured BBC room define the diagnostic's scope.

Each scene supplies five mixture states and three states of each exact
voice. The 66 inputs retain the same six-second PCM16 carrier, measured
room response, delivered level and source clock. Exact component selection
uses benchmark truth. It represents a recovery ceiling available to the
researcher. It supplies no customer-side separator or speaker attribution.
The model, two fixed decision points and speech gate remain unchanged.
The earlier [one-second isolation study](audio-two-voice-isolation-2026-11-15.md)
used a different classifier and view preparation.

## Closed outcomes

| Check | PhonemeDF mixture | PhonemeDF exact voice | Stock mixture | Stock exact voice |
| --- | ---: | ---: | ---: | ---: |
| Adjacent 750 ms brief check, useful source clocks | 0/4 | 0/4 | 0/8 | 0/8 |
| Two-second check, useful source clocks | 0/4 | 0/4 | 1/8 | 1/8 |

The brief check missed all twelve source clocks in both views. Ordinary
isolated voices produced zero flags in 24 observations. The corresponding
ordinary mixtures produced zero flags in 24 target-voice observations
from twelve distinct files. Source-level observations share readers,
waveforms and conditions, so these denominators describe paired evidence.

At a separately preselected best-overlap clock, the synthetic response
exceeded its matched genuine response with both competing-voice states
in 8 of 11 fully admitted comparisons. One of twelve planned comparisons
lacked all required admissions. This known-clock ordering gives a limited
reason to investigate a representation or calibration adapted to room
speech. It supplies neither automatic localization nor a confidence score.

The frozen recovery gate failed. There were zero clear recoveries and
zero new ordinary-voice flags. This candidate stays in development.

## Verification and reuse

An independent implementation replayed all 66 native inputs, 1,056
source/window/admission bindings and 520 distinct classifier inputs. The
scorer made 521 calls including an exact repeat. Native replay verified
5,760,000 exact additive stereo sample identities. The private independent
review also passed seven authored controls. These checks used the same
original sources and a separate implementation.

The companion aggregate includes only cohort counts, whole-protocol
commitments, proof counts and scope. Source names, individual hashes,
paths, exact edit offsets, response vectors and audio remain private.
The standard-library reviewer validates the pinned aggregate, reconstructs
its decision gate and rejects eight authored corruptions.

```sh
python3 -B -S backend/scripts/review_audio_room_voice_recovery_ceiling.py \
  docs/evaluation-panels/audio-room-voice-recovery-ceiling-2026-11-19.json \
  --self-test
```

This public command checks aggregate arithmetic. Exact native and model
reproduction requires the retained original-source custody and pinned
assets. Source terms and render provenance remain in the parent benchmark.
[PhonemeDF release](https://zenodo.org/records/19522979),
[MLS release](https://www.openslr.org/94/) and the
[voice-model publisher](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector)
identify the upstream sources. No new media, weights or generation were
acquired for this diagnostic.
