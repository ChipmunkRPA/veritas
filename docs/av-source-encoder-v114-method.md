# Actual source speech states for native mouth conditioning

Following the successful [authored contracts](native-authored-contract-results-v113.md),
run the exact qualified CPU encoder on all 48 existing source waveforms. This
qualifies reusable source states and routing for later mouth generation. It
does not train or score a detector, inspect confirmation detector outcomes or
measure achieved phonetic alignment.

Keep the v104 sentence/role partition, v105 waveform bytes and v109 native
picture coordinates unchanged: 18 fitting, ten calibration, ten selection and
ten confirmation sources; 3,642 picture coordinates. Use original journal
order in eight sequential batches of six, without source replacement or
response-based selection. Preserve the one short source without a 20-frame
edit and all ambiguous/boundary-support records.

Before each stage, verify the published v110 code/library/asset contract and
successful authored/owned-descendant receipts. Bind the full waveform and
clock journals by hash, then require exact source, role and waveform agreement.
Rehash each mono16k WAV and validate its complete signed-16 header/PCM bytes.
Convert little-endian samples to float32 by division by 32768, without gain
normalization or timing changes. No new weights, models or dependencies are
acquired. The three speech assets retain their publisher-bound Apache-2.0
records; no noncommercial model is introduced.

Use the unchanged `NativeEncoder`: strict encoder-only safetensors load,
CPUfloat32/eager deterministic single-thread inference, fixed log-mel padding,
and independent captures of the four layer inputs/final output. Encode each
source once and the first source of every batch twice, requiring bitwise state
and proof agreement: exactly seven forwards per batch, 56 total. All source
states must remain finite with shape (1500,5,384). No source is treated as
original or synthetic from this execution contract.

For every picture, compare the qualified selector with a separate explicit
tick/layer copy, including exact zeros outside the retained prefix. Reject
unavailable picture/sample support instead of inventing conditioning. Save
each full hidden array privately on the authorized Seagate drive, then load
without pickle and require byte-identical values. Keep individual feature
hashes, source IDs and role pairings private. Record completion receipts,
aggregate routing counts and whole-journal/profile hashes. Array correctness
does not establish narrow temporal locality: whole-input normalization and
unmasked attention remain, as the authored suffix experiment demonstrated.
Generator-conditioning qualification stays false until rendering and response
checks are separately completed.

Use a new exclusive trial/cache and the bytecode-identical v108 resource
runner with isolated globals for this worker, paths and prerequisites. The
prerequisite function is already enforced by `verify()` before admission and
again inside the worker; it requires the qualified v110 monitor and authored
receipts. Keep process groups, descendant sampling every .25 seconds, 900-second
limit, 6-GiB sampled descendant RSS, 6-GiB initial RAM, 3-GiB initial project/archive
disk and 1.5-GiB running reserves unchanged. The reused resource receipt schema
is version 108; the frozen new context and runtime hash identify this execution.
Run outside the sandbox, sequentially after video model work. Samples do not
establish true peaks. Stop on the first failed/incomplete stage, preserving
artifacts; no retry, overwrite, automatic fallback or relaxed admission.

Freeze this code, tests, procedure and input/qualification commitments before
source inference. Publish aggregates and whole-artifact hashes only. This
work changes no serving defaults, calibration, customer confidence, payment
flow or rollback asset. No outreach, GitHub Actions, real charge or GCP action
is part of this test.
