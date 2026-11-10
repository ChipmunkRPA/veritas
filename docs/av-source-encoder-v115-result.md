# Source speech encoder and native routing: actual result

All 48 existing recordings completed the frozen
[v114 source procedure](av-source-encoder-v114-method.md). Eight sequential
CPU batches made exactly **56 encoder forwards**: one per source plus eight
bitwise repeat pairs. All batches exited successfully with no resource guard
or sampling error. These are actual source encoder executions, separate from
the earlier authored tone experiment.

The complete data-only review then verified all stage/response commitments,
source and role bindings, saved array bytes and native time/layer routes.
All **3,642 picture coordinates** matched the separate explicit copy check,
including **479 explicit out-of-prefix tick slots**. The 18 fitting, ten
calibration, ten selection and ten confirmation sources retained their roles.
No authenticity classifier evaluated these sources in this stage, and no
confirmation detection outcome was inspected.

Each private saved array is finite float32 with shape (1500,5,384). Its loaded
bytes reproduce the corresponding captured encoder output. The whole source
state cache remains on Seagate; individual voice-derived features, hashes and
source pairings are not included in the public package. The
[public result](evaluation-panels/av-source-encoder-result-v115.json) contains
aggregates and whole-artifact commitments only.

The review made no neural calls. Its actual 1.20-second execution observed a
maximum self RSS of 95,158,272 bytes between records. Those observations and
the extraction's sampled descendant RSS do not establish true memory peaks.
This is an internal saved-state review, not an independent person's review or
a second neural implementation.

## What this enables and what remains

The native-clock source states are now ready for a separately qualified mouth
renderer. Generation must still preserve source pictures/clocks, demonstrate
response to changed audio conditioning and keep before/during/after support.
Requested conditioning is not achieved mouth synchronization. The one short
source still lacks a valid 20-frame edit, and clock ambiguity remains explicit.

All 48 sources therefore retain **generator-conditioning qualified: false**.
No lip-sync detector was trained or improved by this step; no AI confidence was
created. Whole-input normalization and attention also prevent treating a token's
index as proof of narrowly localized synthetic speech. Next work is native
render qualification, then an articulation detector comparison with ordinary
synchronized speech and other controls under the existing source/role partition.
Serving behavior and rollback assets remain unchanged.

To reproduce the data-only review with the separately retained private source
plan and state cache, use a fresh checkout/context and the exclusive command:

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_av_source_encoder_v115
```

The command refuses an existing review directory; do not overwrite this completed
run. Public authored checks require no media, weights, network or credentials:

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_av_source_encoder_v114.py backend/tests/test_review_av_source_encoder_v115.py
```

Those 21 tests cover signed PCM conversion, changed source bindings, tick/layer
permutations, explicit padding, unavailable support, worker routing and changed
saved state/provenance claims. They test contracts, not detection accuracy.
