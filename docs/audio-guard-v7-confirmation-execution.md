# Audio v7 fresh comparison: fixed execution record

This records the preparation and executable freeze **before the first fresh
classifier forward**. It is not a completed accuracy result. The candidate
remains research only; the serving detector, overall confidence, payment and
rollback behavior are unchanged. GCP remains deferred.

The [prospective design and gates](audio-guard-v7-confirmation.md) are unchanged.
All1080 exact four-second inputs are prepared from120 foreground recordings,
with110 known connected groups, six genuine background speakers, eight unused
noise recordings and four unused music tracks. Conditions are clean,
noise0/5/15, music5/15 and genuine background speech0/5/15 dB SNR.

## Immutable artifacts

- Original execution lock SHA-256:
  `209a0702115374e064baae3d23f0bc03f7933f0366461de6187cdf6116d6798d`.
- Exact variant manifest SHA-256:
  `14ffba2f8bb6b0fe0bb718b69197409544f1171466fa082943dd15d27becc23a`.
- Candidate head/independent guards remain
  `fa1a0faec12fd6926cefb8f7ffe9c9736f517ea89b44a2ff13ab17f4d028c408`;
  current serving calibration remains
  `912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf`.
- The [public input capsule](evaluation-panels/audio-v7-confirmation-inputs.json.gz)
  SHA-256 is
  `7b2f7538a975d306f5d2b64a39fbae2cd511257d4c49fb8595564b0107a1e157`.
  The [small execution receipt](audio-guard-v7-confirmation-execution.json)
  binds these records. The capsule includes every recipe, original small head/
  projection parameters, baseline calibration, numerical runtime, exact code
  and commercial checkpoint commitments, and source/consumed-parity receipts.
  Full source audio, transcripts, private paths and pretrained checkpoints
  are excluded. Source references and terms remain separately required.

The lock pins all inference/decision/metric/gate code and imported scientific
helpers. It records Python, NumPy, SciPy, scikit-learn, torch, transformers,
pyloudnorm, soundfile and the native CPU float32 Mac runtime. Four threads are
used. No parameters or directional points are selected from fresh outcomes.

## Input and service-path checks

All138 source recordings were checked through production decoding. Separate
mixture reconstruction verified1080 files and69,120,000 sample values, plus18
sampled production FFmpeg decodes. The largest realized SNR deviation is
0.000143 dB. Canonical16-bit WAV encoding avoids changing timestamp metadata
in float-WAV headers; its maximum quantization error is exactly half a16-bit
sample step. Original audio is preserved separately. Each condition uses the
same source-specific peak gain across all nine variants, with no clipping.

The new runner reproduced thirteen consumed cases: both truth classes in all
six historical conditions, plus one unassessed input. Original neural outputs,
current guarded probabilities/decisions and v7 probabilities/decisions matched
exactly. The initial passing receipt is retained locally; after completing
imported-helper/runtime bindings, the same consumed cases were rerun and again
matched exactly. This did not score any fresh source or change a parameter.

The baseline invokes the actual app's frozen calibration. Both arms use the
same original-view Silero gate/windows and Pella representation. A bandpass
probe is shared only when either original tendency needs it; the service uses
its existing raw-response rule, while v7 uses its own calibrated probability
rule. A failed probe withholds original tendencies. Missing speech produces
two inconclusive outcomes and remains in the denominator.

## Reproduction and interpretation

With the declared numerical dependencies, review the public preparation
without acquiring audio or pretrained checkpoints:

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.reproduce_audio_confirmation_v7 docs/evaluation-panels/audio-v7-confirmation-inputs.json.gz /tmp/audio-v7-confirmation-input-review.json
```

Keep `audio-v7-fresh-sources.json` beside the capsule; its exact hash is checked.
The review checks each source/group/truth binding, all source/condition/sample
clocks, the original candidate head artifact hash, known background references
and complete audit/parity receipts. It does not regenerate waveforms, rerun
neural networks, establish label truth independently or recruit an outside
reviewer.

The [local preparation replay](evaluation-panels/audio-v7-confirmation-input-replay.json)
passed. The targeted source/calibration/scoring/public-review suite passed44
tests, including five altered-artifact rejection cases. New source and expanded
capsule secret scans were clean. These are local checks, not outside testing.

The [scoring runner](../backend/scripts/score_audio_confirmation_v7.py) resumes
only an exact completed prefix under the same immutable lock. Its
[numerical reviewer](../backend/scripts/review_audio_confirmation_v7.py)
requires all1080 terminal outcomes and reconstructs probabilities, guards,
stability, complete operational counts and the predeclared gates. Proper
scores use common assessed inputs; missing/failed/inconclusive outcomes stay
in operational denominators. Connected-group bootstrap order is deterministic.

Even a passing comparison applies only to these short speech windows. It
would not establish whole-recording calibration, interval localization,
per-customer accuracy, independence of pretrained data or outside validation.
Local adapter/evidence/PDF/JSON/paid-flow/rollback verification would still be
required before enabling the new calibration. A failed comparison preserves
the current service and makes these sources consumed development material.
