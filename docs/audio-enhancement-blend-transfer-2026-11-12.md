# Check short speech edits on enhanced and partly restored audio

The comparison tests whether a speech response survives when part of the
original recording is put back into an enhanced view. It uses two dependent
views of the same samples and the same voice classifier. It does not infer
the source of the background sound or produce an authorship confidence.
[Existing enhancement and asset terms](audio-nemotron-noise-transfer-2026-11-12.md)
· [Earlier long-audio study](audio-reduced-background-localization-2026-11-12.md)

## Frozen comparison

Revisit eligible dense three-second windows on the 98 exact deliveries from
the focused-window study. Generate an enhanced view on the original sample
clock with the pinned local assets. Reapply speech activity and the previously
fixed enhancement point. For an above-point response, average the original
and enhanced waveform at equal amplitude and requantize to 16-bit PCM. The
second arm requires speech activity and an above-point response on both
views. No original-leaning window can be promoted. Either arm needs adjacent
AI-like windows to add an interval and keeps all existing focused ranges.
[Source and edit clocks](audio-focused-window-transfer-2026-11-12.md)

The blend is a fixed sensitivity test. It shares the enhancement point as a
conservative requirement and has no separately fitted customer operating
point. The original records have prior project exposure. The two conditions
represent two background recordings, with repeated voices across conditions.

## Held result

| Held delivery | Focused useful edits | With enhanced audio | With support from both views | Genuine flags, enhanced / supported |
| --- | ---: | ---: | ---: | ---: |
| Gain-only | 13/18 | 13/18 | 13/18 | 0/12 · 0/12 |
| Rain, −5 dB | 3/18 | 3/18 | 3/18 | 1/12 · 0/12 |
| Train, −5 dB | 0/18 | 2/18 | 0/18 | 1/12 · 0/12 |

The enhancement found a new location in each of two train-masked files,
from LJJETS and YourTTS. Two additional train files and three additional
rain files received intervals without a useful edit location. Across the
two scenes, the enhancement added 21.411 seconds of marked genuine time.
Of that, 4.500 seconds came from entirely genuine files and 16.911 seconds
from genuine sections of edited files. None of the eight background-only
excerpts received a flag.

Speech activity admitted 224 of 250 enhanced windows. The 45 above-point
responses were replayed as blends. Of those, 43 passed speech activity and
zero remained above the point. The stricter rule lost both train gains and
all new false flags. The enhancement results do not qualify either arm to
serve as a localization or confidence upgrade. Native replay verified
all 250 enhanced window hashes and all 45 blend hashes.
[Result and exact window commitments](evaluation-panels/audio-enhancement-blend-transfer-2026-11-12.json)

## Replay

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_enhancement_blend_transfer review \
  docs/evaluation-panels/audio-enhancement-blend-transfer-2026-11-12.json
```

For native replay, set `held` to the retained PCM from the focused study and
`run` to a new private directory. The two local model paths are the pinned
enhancement checkpoint and the existing Apple Silicon dependency overlay.

```sh
export PYTHONPATH=backend VERITAS_AUDIO_DETECTOR_MODE=candidate VERITAS_BILLING_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_enhancement_blend_transfer enhance "$run" \
  --held "$held" --model "$model" --overlay "$overlay"
backend/.venv/bin/python -B -m scripts.audio_enhancement_blend_transfer score "$run" --held "$held"
backend/.venv/bin/python -B -m scripts.audio_enhancement_blend_transfer review "$run/receipt.json" \
  --held "$held" --native-run "$run"
```
