# Audio v8 prospective execution freeze

**Superseded before fitting.** A later publisher-identity audit found six voice
groups crossing the intended data roles. DuplexGen's cloned-voice IDs refer to
LibriSpeech speakers; distinct dataset prefixes had hidden those links. The
extraction was stopped after1882 records, before any candidate fitting or fresh
scoring. No v8 replacement or accuracy improvement is supported. The original
freeze, waveform checks and receipts remain preserved. See the
[identity correction](audio-source-identity-v9.md); the metadata-only review below
did not establish complete cross-corpus speaker independence.

6 November 2026. This record captures the state **before v8 neural extraction or
project-data fitting**. It is not a performance result. The current app's audio
calibration and overall confidence remain unchanged.

| Artifact | SHA-256 |
|---|---|
| Original source/identity plan |`3b6d406380c4b5887f352dff8045071d4915926cc627459df3f314f7105a26e0`|
| Prepared sources and separate backgrounds |`018539600224f87515bc0da43163a931b3c0f639d615e8b199f51b7a1284f6b3`|
| Exact4284 consumed-role variants |`911a865638c1a49f367071a1a0eaebcf9c196b9dc2ae64b3f965c92d81e79515`|
| Exact1080 reserved fresh variants |`6b97000fa9c458f919511451abfbe049ce3aa4adca142d86050cac296310367d`|
| Development method/input/code/runtime/weight lock |`94528b1f27633d00ab82d1485542874d7a7e9200b08ac364130864adb4fe527e`|
| Public compressed source/recipe capsule |`ba58b6646be0d9ce36f8675d80fff3ea146ced3b767882d86bcd02cfa53af0f3`|

The [protocol](audio-multisource-v8-development.md) defines the fixed learner,
independent calibration, stability checks, comparator and16 advancement gates.
The [public capsule](evaluation-panels/audio-v8-source-plan.json.gz) is1,521,389
bytes (8,972,258 expanded JSON bytes). It contains596 foregrounds,66 backgrounds
and5364 recipes. Only120 foregrounds are reserved fresh; all476 others have
already been consumed in earlier experiments. V8 must not call those sources a
new held-out test.

## Checks completed at the freeze

- All662 original source-byte and full production-PCM identities are distinct.
  Actual four-second prefixes, known speakers, full/script-opening text and
  conservative shared-prompt connections stay within assigned roles. Foreground
  connected-group counts are174fit,84calibration,106development and120fresh.
- All78 newly reserved genuine speakers were materialized from complete,
  publisher-checksummed original Parquet files. The28 recordings collected before
  dataset-server rate limiting retain the exact selected recording and full/
  prefix PCM. Four ineligible candidate recordings remain recorded; reserved
  identities and original eligibility rules were preserved.
- Initial background preparation failed because only18 cached unused noise
  recordings were at least eight seconds. Preserve its code/log. A pinned,
  checksummed original Parquet row-group1 extension supplied100 public-domain
  subset originals,52 meeting the duration rule. No source was chosen by a
  classifier response, and no duration rule was relaxed.
- Model-free public review checked every role, source/condition/background
  binding, sample clock, common peak gain and40 implementation-file pins.
  All five real-capsule alteration checks were refused. These are metadata
  checks, not source-media or neural-output reconstruction.
- The39 focused v8 tests and221 selected audio/calibration/source/reproduction/
  independent-contract regressions passed. Lint passed. A privacy test caught
  normalized text in an early unpublished export; final public connection
  fields are hashes. The software and expanded capsule undergo separate secret
  scans. Fingerprint-field alerts were independently recomputed against source
  metadata before an exact field/path exception; no broad secret exception.

## Remaining work

Publish this freeze before extraction. Extract all4284 fitting/calibration/
development inputs with the locked CPU float32 runtime and verified existing
commercial-permission weights. No fresh rows are accepted by the development
runner. Fit only fitting roles, save those parameters, then fit the independent
calibration guards and compare all1422 development inputs with the actual current
guarded service. Preserve errors and unassessed cases.

Actual source/mixture/native-forward/probe replay and numerical refitting remain
required. A passing development comparison permits a separately locked fresh
comparison only; adoption also needs fresh results and paid-flow/evidence/export/
rollback verification. A failed gate stops advancement. No actual GCP resources,
model default, overall confidence, billing or authentication changes are part of
this freeze. No external reviewer has validated it; the public resources enable
review without claiming that it has occurred.

## Source reconstruction completed after publication

The original freeze and its40 implementation pins remain unchanged. A separate
read-only source audit, added after extraction started, independently rebuilt all
5364 exact mixtures from all662 original recordings. Float payloads, complete WAV
files and production PCM matched; all72 predetermined role/class/condition native
decoder checks matched. It does not inspect learned responses or change inputs.
The [complete audit receipts](evaluation-panels/audio-v8-input-audit.json.gz) retain
every case. This is actual source reconstruction, separate from the earlier
metadata-only reviewer.

The [offline helper](../backend/scripts/rebuild_audio_multisource_inputs_v8.py)
then rebuilt all5364 inputs using **only the public capsule and a cache of complete
original recording bytes**. It uses no private manifests, source text, provider
IDs or neural models. Its [complete receipts](evaluation-panels/audio-v8-public-reconstruction.json.gz)
contain5364 successes and zero missing/failed backgrounds. Their original JSONL
SHA-256 is`a2638450ceade866c6714fa7097a7df6efc76029ec6c8fb17ff458224031d9ba`.
The43 focused v8 tests passed, including original-cache corruption/path rejection,
actual SNR preservation and source-location validation. The earlier221 selected
regressions remain the freeze's broader check.

The original capsule omitted direct URLs for141 older source entries. An additive
[location supplement](evaluation-panels/audio-v8-source-locations-supplement.json)
maps those same original-byte hashes to publisher files. Its six referenced
archives were checked against their publisher MD5/SHA-256; direct recording assets
must match the frozen original's byte count and SHA-256. No case, role, selection,
method, parameter or capsule byte changes. The supplement's SHA-256 is
`6b4675428eca594d1e0b2c9967fb9f344bb27a6fe207768fbbbb6aa4550724fb`.

Acquire complete originals separately under their per-source terms. Use the
capsule's Parquet row coordinates or archive-member-name digest and original-byte
digest, plus the supplement where needed. Place each original in a separate cache
as`<sha256>.audio`. From the repository root:

```sh
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/rebuild_audio_multisource_inputs_v8.py \
  docs/evaluation-panels/audio-v8-source-plan.json.gz \
  /path/to/separately-acquired-source-cache /path/to/new-output-directory \
  --capsule-sha256 ba58b6646be0d9ce36f8675d80fff3ea146ced3b767882d86bcd02cfa53af0f3
```

The helper downloads nothing and preserves every planned receipt on source
failures. Existing output is refused. A decoder or numerical difference must be
reported rather than silently replacing a source. These successful local replays
are not independent human review or evidence of better detector accuracy. Learned
extraction is still a separate ongoing stage; fitting/calibration/development and
fresh confirmation remain required before any replacement.
