# Noise-aware speech experiment: prepared before source inference

## Completed result

Both fixed candidates are rejected. All4,248 extractions and the corrected
1,449-input comparison completed; the original34-record initialization failure
is retained. Combined features improve noise/music but fail generator and
background-speech retention/usefulness. All593 native replays, numerical review
and both exact refits pass. Empty-cache public-only replay reproduces the complete
feature receipt and both fits, with12 actual-packet alteration rejections.
The [complete result and reproduction](audio-wavlm-v13-development-evaluation.md)
records every condition and limit. No reserved confirmation or service change.
Preparation and the original execution chronology below are retained.

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

The freeze was published and remotely verified as `45e8ff9` before any project
speech forward. Extraction then completed all4,248 records:4,186 assessed,
62 unassessed and zero failures. Both fixed heads converged on fitting sources
and their own independent calibrations were saved. Comparison has no complete
result; the initialization failure and continuation are recorded below.
These execution facts do not establish an accuracy gain.

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
PYTHONPATH=backend backend/.venv/bin/python -m scripts.execute_audio_wavlm_v13 \
  compare "$research_root/lock.json" "$research_root/features.jsonl" \
  "$research_root/trained/head.json" "$research_root/responses-callable.jsonl"
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_wavlm_v13 \
  review "$research_root/lock.json" "$research_root/summary.json" \
  --receipt "$research_root/features.jsonl" --head "$research_root/trained/head.json" \
  --responses "$research_root/responses-callable.jsonl"
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

## Additive verification prepared before development outcomes

`review_audio_wavlm_v13.py` separately reconstructs linear probability arithmetic,
fitting-only normalization, worst-group/truth calibration, raw/mapped stability
and complete operational counts. It uses the shared declared statistical
bootstrap/gate implementation and original learner for optional exact refits.
Its native mode replays the union of99 predeclared cells and every required
filtered probe, including original model/status/window parity. The helper does
not change the69 frozen method files or any acceptance rule. Eight hand-calculated
and invalid-input predictor checks passed; no native/numerical result is implied
until those stages actually finish.

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.review_audio_wavlm_v13 \
  numerical "$research_root/lock.json" "$research_root/features.jsonl" \
  "$research_root/trained/head.json" "$research_root/responses-callable.jsonl" \
  "$research_root/numerical-review.json" --summary "$research_root/summary.json" --refit
PYTHONPATH=backend backend/.venv/bin/python -m scripts.execute_audio_wavlm_v13 \
  native "$research_root/lock.json" "$research_root/features.jsonl" \
  "$research_root/trained/head.json" "$research_root/native-review.json" \
  --responses "$research_root/responses-callable.jsonl"
```

## Recorded execution initialization correction

The first comparison stopped at input34 with a retained RuntimeError: both the
research CLI and the original-model loader initialized PyTorch inter-op threads.
PyTorch permits that process initialization only once. A minimal duplicate-call
reproduction confirms the cause; the same input and exact frozen validator pass
through the callable functions without the redundant CLI initialization.

`execute_audio_wavlm_v13.py` calls those unchanged functions in a dedicated
process, setting the declared four intra-op threads and leaving inter-op ownership
to the original-model loader. The [execution supplement](evaluation-panels/audio-v13-thread-ownership-supplement.json)
binds the launcher, original69-file freeze, complete feature receipt, both original
fitted/calibrated objects and failed-prefix digest before continuing. Use a new
`responses-callable.jsonl` journal; preserve the original failed34-record journal.
No source, feature, fitting, calibration, filtering, stability or gate changes.
No detection result was selected or rescued by this initialization correction.
