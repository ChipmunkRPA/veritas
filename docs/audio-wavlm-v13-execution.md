# Noise-aware speech experiment: prepared before source inference

6 November 2026. Prepared a different frozen backbone after the fixed-layer
experiment failed useful-detection requirements. The [prospective protocol](audio-wavlm-v13-protocol.md)
fixes both arms, extraction, fitting/calibration and all original sixteen gates.
This preparation establishes no detection improvement or service replacement.

## Verified source and preparation

The anonymous official download matches its publisher LFS digest exactly:
377,617,425 bytes. The [source receipt](evaluation-panels/audio-v13-wavlm-source-review.json)
records pinned model/config/preprocessing/license references. The model card links
CC-BY-SA-3.0 weight terms; the separate unilm code is MIT. Both notices are retained
locally. No checkpoint or training corpus is added to the repository.

Weights-only inspection found no unexpected pickle globals. Standard installed
Transformers construction loaded all248 tensor entries strictly, with94,381,936
parameters and zero trainable parameters. No custom remote code was executed.
Two forwards on deterministic sine-plus-noise, containing no speech, reproduced
all4,608 features exactly, preserved input bytes and verified the199-frame clock.
That smoke checks the execution path, not detection accuracy or customer latency.

All58 selected tests passed, including25 new boundary/learner tests and existing
layer/role/public-contract tests. They cover padding exclusion, population
statistics, selected-layer ordering, source shape/clock/digest validation,
unassessed preservation, fitting/calibration isolation, independent missing-group
guards and rejection of changed prospective commitments. Lint and whitespace
checks passed. Synthetic tests do not replace native speech replay.

## Frozen experiment

All4,248 original waveforms were rehashed; original corrected roles and all62
unassessed inputs remain. The private lock SHA-256 is
`5efc68aef52a5d45459da94321f85cefea77af8c5d3a22f090f2d2d9211ed54f`.
The [public freeze](evaluation-panels/audio-v13-wavlm-freeze.json) binds69 project
files, five installed implementation files, original baseline weights/runtime,
new pretrained assets, source/recipe/confirmation digests and the99-cell native
plan. Private roots are excluded. A separate [preflight](evaluation-panels/audio-v13-wavlm-preflight.json)
accepts the unchanged actual packet and rejects13 deliberate alterations.
No speech-derived feature vectors are in this prospective metadata package.

The first freeze attempt stopped because the destination directory did not exist;
it wrote no lock or feature journal and ran no project forward. Created the empty
directory and repeated the unchanged preparation successfully. Both logs remain
locally. No method, source or acceptance rule changed to obtain that success.

## Ordered execution

Publish and remotely verify the code/prospective metadata before the first project
source forward. With pinned original research inputs and checkpoint assets
provisioned, and the commercial/offline cache settings configured:

```sh
research_root=/path/to/new-audio-v13-wavlm
mkdir "$research_root"
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_wavlm_v13 \
  freeze /path/to/audio-v9 "$research_root/lock.json" --weights-root /path/to/wavlm-assets
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_wavlm_v13 \
  extract "$research_root/lock.json" "$research_root/features.jsonl"
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_wavlm_v13 \
  train "$research_root/lock.json" "$research_root/trained" --receipt "$research_root/features.jsonl"
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_wavlm_v13 \
  compare "$research_root/lock.json" "$research_root/responses.jsonl" \
  --receipt "$research_root/features.jsonl" --head "$research_root/trained/head.json"
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_wavlm_v13 \
  review "$research_root/lock.json" "$research_root/summary.json" \
  --receipt "$research_root/features.jsonl" --head "$research_root/trained/head.json" \
  --responses "$research_root/responses.jsonl"
```

The fitting stage refuses incomplete/failed extraction and saves fitting objects
before calibration. Replays can resume only a validated exact prefix; failures
remain visible and cannot be erased by restarting. An observation timeout does
not mean a process stopped. Both model-specific calibrations preserve original
scope, internal research points, dual stability and all sixteen adoption gates.

After complete development, reconstruct both fits and all operational arithmetic,
then replay the99 native cells plus every required filtered probe before any
separately frozen fresh confirmation. Failure rejects the arm without scoring
the reserved set. Passing development alone changes no service or confidence.
Independent outside human reviewers remain zero. Image/video/writing work and
outside testing remain unfinished; GCP stays last.
