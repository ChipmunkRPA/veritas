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

All **2,088 fitting/calibration cases have completed** across 24 bounded stages,
with no guard, sampling error or resource-limit observation. Extraction took
2,956.14 seconds total. Maximum sampled child-tree RSS was 4,807,589,888 bytes;
minimum observed available RAM was 3,731,980,288 bytes. These are sampled limits,
not guaranteed peak measurements. The
[resource receipt](evaluation-panels/audio-combined-confidence-v84-fitcal-resources.json)
retains the figures separately from any predictive-performance claim.

The two new combinations finished fitting and their own separate calibration
maps. The global method used 999 assessed fitting cases and 1,008 assessed
calibration cases. Both measured contexts have calibration support: low context
641 cases, high context 367 cases. The trained coefficients and group/source IDs
remain private; the [training receipt](evaluation-panels/audio-combined-confidence-v84-training.json)
publishes aggregate support and whole-artifact hashes only. A separate fitted-
map arithmetic check reproduced all 2,007 assessed probabilities within 2.23e-16,
kept 81 unassessed outputs missing, and verified predictions are unchanged when
truth/provider/condition/source metadata are omitted. This
[wiring check](evaluation-panels/audio-combined-confidence-v84-fitted-wiring.json)
uses fitting/calibration data and does not establish held-role reliability.

Selection and final-confirmation outcomes remain unused. The exact selected
methods must be published before final forwards. There is no completed fresh
selection/final detection or new calibration-quality result yet.

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

The same reviewer subsequently completed all 24 fitting/calibration chunks:
**2,007 assessed cases replayed, 81 unassessed cases preserved**. The
[complete fit/cal replay receipt](evaluation-panels/audio-head-replay-v85-fitcal.json)
extends the prefix result; it remains an arithmetic/data-integrity review,
not an accuracy measurement or outside human participation.

All 20 experiment/reviewer checks and lint passed locally. Raw audio, source
IDs/pairings, individual fingerprints, compact/full response records, features
and learned coefficients remain private. Aggregate metrics, errors, coverage
and limitations will be reported after selection and final confirmation. No
workflow, outreach, real payment or GCP work is part of this execution.

The [supplemental v86 diagnostics](audio-fusion-diagnostics-v86-protocol.md)
were fixed during fitting/calibration extraction, before selection/final
forwards. They preserve every condition-by-family cell without changing the
primary method or selection rule. The
[background-exposure histograms](evaluation-panels/audio-background-exposure-v86.json)
confirm overall truth-class balance but show uneven exposure within individual
AI families. Also, class-only error/bin summaries must not be interpreted as
standalone probability-calibration tests. Five additional checks verify joint
counts, missing outcomes and background-role/exposure handling. These are
secondary descriptive diagnostics, not a new performance or reliability result.

A [parallel shared-reference calibration alternative](audio-common-reference-v87-protocol.md)
was fixed after these development/calibration results and before any held-role
forward. It preserves the existing context score fits and changes only whether
reference weighting is balanced within each context or shared across contexts.
It reuses development/calibration data rather than claiming new independent
calibration data. Primary v84 selection stays intact; both fixed choices must
precede final forwards. The same held recordings compare both analyses, so they
are not two independent confirmation cohorts.
