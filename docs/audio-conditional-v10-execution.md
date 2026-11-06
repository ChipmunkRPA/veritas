# Conditional audio experiment: preparation and execution

6 November 2026. Prepared the next comparison after v9 failed five requirements.
This [prospective protocol](audio-conditional-v10-protocol.md) separates the
nonlinear learner from its added waveform-quality inputs, with a quality-only
nuisance check that cannot be adopted. No new project-data model has been fitted
at this freeze. The current app, confidence, payment/auth and GCP setup are unchanged.

## Prepared inputs and commitments

All 4,248 original quality measurements completed with zero failures. They use
the unchanged v5 descriptor code on exact original production PCM, with the same
2.5–4second supported shape and four-second benchmark windows. All 402 retained
development probes also have separately recomputed filtered-waveform quality
and matching filtered-PCM digests. Quality measurements do not alter speech
assessment status; the original 62 unassessed cases remain unassessed.

The original neural features, fitting-only projection and baseline/probe results
remain the exact published v9 artifacts. No waveforms, sources, roles, groups,
conditions or reference verdicts were substituted. The 1,080 reserved confirmation
inputs remain fixed and unscored. These are consumed development inputs, not a
new independent accuracy panel.

- Private lock SHA-256: `263419ee9743939beaf767b519ed64ea935bf6768312ffd927caa47cebbfb7c9`.
- [Public freeze](evaluation-panels/audio-v10-conditional-freeze.json) SHA-256:
  `bc5711f4b2f3fdbc68509d7f89c7e7647179fcbe5056e9fa817352b7e2b977a8`.
- [Exact quality JSONL](evaluation-panels/audio-v10-quality-inputs.jsonl.gz),
  compressed629,961bytes, SHA-256
  `1b9f5cc344a607831a4c610be0a837e04adedbbe3c8ddb5d535db30ac3be2711`.
  Expanded1,641,169bytes SHA-256:
  `e8b7d025cf9e6c5798d4a6b313486d3a4ccfac7c2fb47ba218957b9622b0bc8e`.

The freeze binds54 project implementation files, twelve installed native/Python
learner files, the exact runtime, verified commercial checkpoint bytes and the
BSD-3-Clause learner-license digest. The public copy excludes private filesystem
locations; the private path-bearing lock has its own distinct digest.

## Verification before fitting

All96 selected regression tests passed, including numerical-tree/native parity,
JSON roundtrip, raw-coordinate monotonicity, metadata-free predictors, independent
roles, unassessed inputs and fixed candidate selection. These use synthetic
fixtures and do not measure detector accuracy. Ruff and whitespace checks passed.

The [model-free public preflight](evaluation-panels/audio-v10-conditional-preflight.json)
checked every source/role/hash/finite-descriptor binding and all54 project code
pins. Seven deliberate changes to the actual public freeze/measurement set were
refused in the [alteration receipt](evaluation-panels/audio-v10-conditional-tamper-review.json).
This is local metadata/code review, not independent human review or a waveform
recomputation by an outside party.

Without a model download or private manifest, from the repository root:

```sh
python3 backend/scripts/export_audio_conditional_v10.py review \
  docs/evaluation-panels/audio-v10-conditional-freeze.json \
  /tmp/new-audio-v10-preflight.json --project .
```

Keep the original `audio-v9-identity-plan.json.gz` catalogue beside the new
resources. The original v9 numerical capsule provides all frozen representations
for later numerical reproduction; it is referenced, not duplicated here.

## Next stages

Publish the freeze before project fitting or new neural forwards. Then fit only
the214 fitting originals and calibrate each arm on the97 separate originals.
Compare all1,449 development cases against the exact served baseline. Report
both candidates and the nuisance check, all sixteen unchanged gates and the
predeclared arm selection. JSON/native prediction parity covers all assessed
originals without using held labels for learning.

Preserve every failed or unassessed outcome and actual probe quality. No new
candidate becomes default from development alone. Source/quality/native/probe/
numerical/refit audits and a passing development result precede any separately
frozen confirmation. Paid-flow/evidence/export/rollback and any overall-confidence
admission remain later requirements. Image/video/writing and real outside human
testing remain unfinished parts of the wider improvement objective.
