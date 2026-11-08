# Reserved calibration producer: v28 preparation

The v26 training sequence remains active. This preparation records a tested
calibration producer and resource monitor; **no reserved neural inference is
admitted or has occurred**. It does not contain detection-accuracy results.

The producer refuses to score without an execution supplement binding complete
training, the final learned checkpoint, unchanged roles, private output paths
and the exact producer/readout implementation. That supplement must be published
and remotely verified before the first reserved forward. Its preparation and
complete training-chain audit still need to be executed after training finishes.

## Implemented contracts

- A training-chain auditor requires all 40 terminal resource profiles, the
  exact 1,278 prescribed updates, unchanged learned-state links, all retained
  checkpoint hashes and update/resource journal hashes. Failed, guarded,
  missing, reset or held-out training phases cannot authorize calibration.
- Only the 32 calibration groups/39 photos are eligible. Preserve four views
  per photo and all 156 views per arm/format. Split them into six complete
  24-view phases and one 12-view phase; photo quartets stay together.
- Candidate prediction loads only the final learned adaptation into the pinned
  native image-only graph, verifies its parameter hashes and uses full-source
  geometry. Native and JPEG75 baselines use the actual pinned current checks.
  The JPEG baseline requires its existing MPS path; no CPU substitution or
  replacement checkpoint is silently introduced.
- Encode JPEG75 exactly once using the frozen training-format convention.
  Bind encoded bytes, decoded RGB, source dimensions and the precompression
  truth PNG. Negatives explicitly carry zero AI truth; no re-differencing a
  JPEG against the original source creates new AI labels.
- Save exact float32 raw response fields privately with file hashes, source
  coordinates and candidate patch/input bindings. Reload every saved map and
  require exact equality. Repeat the first input of each arm/format exactly.
  A complete calibration run would make 624 unique-view model calls plus four
  exact-repeat calls; those calls have not happened.
- Record attempted and completed calls. A model-setup failure marks all views
  planned for that phase unavailable. An input/inference failure retains a
  failed row and stops the phase. Missing downstream rows remain unchecked,
  not clear or original. Never overwrite outputs or skip a failed predecessor.
- The separate monitor records exact resource observations, limits each phase
  to 900 seconds/6 GiB sampled tree RSS, requires initial headroom and preserves
  1.5 GiB available RAM and internal/external disk reserves. No implicit retry.
  Sampled worker RSS does not guarantee peak usage or separately account for
  every Metal allocation; the RAM reserve remains independently observed.

## Local verification and remaining work

Fourteen local tests and lint passed. Tests cover complete cohort/chunk counts,
format/coordinate parity with the frozen training encoder, incomplete-training
and missing-publication refusal, model-unavailable denominators, exact map
roundtrip/first-input repetition, and refusing to start without memory headroom.
The writer tests use explicit test doubles. They are not neural inference,
model-quality tests or evidence that a detector finds edits.

The [preparation record](evaluation-panels/image-regional-inference-v28-preparation.json)
binds this source, monitor, tests and document. Preserve them as this version;
any correction needs an explicit successor. The later execution supplement must
also include the final-chain proof and complete runtime/asset identities.

After all v26 training phases complete, audit their actual final state under a
bounded local data phase, publish and verify the full execution supplement,
then run candidate/baseline calibration sequentially. Compute the fixed v27
map-policy readout afterward. Point, diagnostic and confirmation execution
requires a subsequent supplement binding the selected policy; this producer
does not authorize those cohorts. Probability calibration, broader generators,
full app integration and serving admission remain outstanding.
