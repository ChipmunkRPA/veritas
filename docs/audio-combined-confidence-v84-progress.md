# Combined audio confidence study: execution status

The [v84 procedure](audio-combined-confidence-v84-protocol.md) was published
before its study neural calls, with the complete source/mixture hashes,
commercially permitted model assets, unchanged component functions, both new
candidate calibration methods and fixed evaluation/selection procedure.
The [published freeze](evaluation-panels/audio-combined-confidence-v84-freeze.json)
is the exact setup used locally. No serving default has changed.

All **18 consumed native reproduction cases passed**: raw scores, complete
representations, current decisions and complete old candidate decision details
match their recorded references exactly. Measured flatness repeats exactly.
The bounded stage passed in 42.02 seconds, with a sampled child-tree RSS maximum
of 4,614,176,768 bytes and no guard, sampling error or resource-limit observation.
This means eighteen reproducible responses, **not eighteen correct detections**.
See the [parity receipt](evaluation-panels/audio-combined-confidence-v84-parity.json).

New-source extraction is running sequentially through the fitting and calibration
roles. The dispatcher stops at the first failed stage and does not retry or
overwrite existing results. It cannot start selection or final-confirmation
chunks. Both new combinations must finish independent fitting/calibration before
selection, and the exact selected method must be published before final forwards.
There is no completed fresh detection or new calibration-quality result yet.

A separate numerical reviewer replays the first seven completed fitting chunks
without neural model calls. Across **609 controlled cases**, it preserves all
29 unassessed cases and verifies feature hashes, original input sample hashes
and four-second clocks for the 580 assessed cases. It reconstructs the raw
original head from its saved representation within 8.75e-8 probability, and
reconstructs the serving calibration and three retained learned-head outputs
exactly in this prefix. The reviewer uses separate arithmetic rather than the
detector prediction functions. Its three authored checks also reject altered
scores and changed/non-lossless feature payloads.

This replay verifies score wiring and recorded data integrity. It does not
adjudicate truth, re-run stability decisions, measure accuracy, establish new
confidence reliability or involve an outside human reviewer. It covers a fixed
prefix of fitting data, not the full 4,176-case study. See the
[aggregate replay receipt](evaluation-panels/audio-head-replay-v85-prefix.json)
and [reviewer](../backend/scripts/replay_audio_scores_v85.py).

All 20 experiment/reviewer checks and lint passed locally. Raw audio, source
IDs/pairings, individual fingerprints, compact/full response records, features
and learned coefficients remain private. Aggregate metrics, errors, coverage
and limitations will be reported after selection and final confirmation. No
workflow, outreach, real payment or GCP work is part of this execution.
