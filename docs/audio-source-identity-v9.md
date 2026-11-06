# Audio source identity correction before fitting

6 November 2026. The v8 experiment is **superseded for speaker separation**.
No v8 candidate was fitted, scored on fresh confirmation, enabled in the app or
added to overall confidence. The current serving calibration remains unchanged.

## What the wider audit found

The earlier v8 checks accounted for540 source references in the selected v3/v7
pool. That was not the complete historical evaluation pool: the original audio
and v2 studies also used264 foregrounds, including40 Kokoro recordings. The new
audit covers all868 foreground/background references in the audio/v2/v3/v7
manifests, plus current sources:1054 distinct original recordings in total.
All1054 original byte/production-decode checks passed. The reserved recordings
share no known original bytes, non-silent2–4second decoded prefixes, full texts
or eight-word script openings with those historical originals. No reserved text
matched the separately declared LJ Speech prompt bank.

New recordings do not necessarily mean new voices. All12 reserved Kokoro
profiles, two cloned LibriSpeech identities and two genuine LibriSpeech identities
had appeared in earlier project evaluations. These remain fixed recordings, but
must not be advertised as an entirely unseen-voice panel. Absence from the
pretrained detector's training data remains unverified.

The more serious defect was a missed cross-corpus identity join. The publisher's
[generation code](https://github.com/duplexgen/duplexgen-code/blob/main/tts_render/convert_spoken.py)
assigns `user_prompt_speaker_id` from its LibriSpeech speaker pool. Its
[data-license/provenance notes](https://github.com/duplexgen/duplexgen-code/blob/main/docs/DATA_LICENSES.md)
also describe those LibriSpeech prompts. Treating `duplex-prompt:<id>` and
`librispeech:<id>` as unrelated made the old checks insufficient. The fetched code
was verified against Git blob`dfb183bad2ba648dbf021f004c71bafb048cabb4`
(SHA256`e282bd3ae44fe42d05a4ab43a9f3ed8ed23f20be53d5f81e23a49748c42fb85d`).
The mapping is publisher metadata evidence, not biometric verification.

Nine known speaker groups were split across statistical groups; six crossed data
roles. None shared a foreground identity with a selected background speaker.
The original metadata-only reviewer passed because it compared literal namespaced
keys. Its40 implementation pins and recipe checks still verify their stated
bytes, but do not establish complete identity independence. The actual source/
mixture/WAV/PCM reconstruction remains valid and is preserved.

## Preserve the failed design and repair before fitting

Stopped the exact v8 extraction process before fitting, retaining all1882 terminal
records:1857 assessed and25 unassessed, zero extraction failures. The original
receipt SHA256 is`68673bfb01c3e5b80609299c4455f3e436d723312f328536bacf95fbd2a7e6f4`.
No model responses select the repair, and the original freeze/input capsules are
unchanged. Do not complete or fit the old role plan.

A separate [retrospective v7 check](evaluation-panels/audio-v7-identity-lookback.json)
compared its120 confirmation recordings with its actual fitting/calibration
sources using the same publisher identity mapping. It found zero known parent
speaker overlaps. The previous v7 outcome counts and rejection remain unchanged;
this is a scoped metadata check, not a biometric or pretrained-corpus guarantee.

The corrected plan canonicalizes known LibriSpeech parent IDs across genuine and
cloned sources. It also normalizes Kokoro profile namespaces, keeps VCTK genuine/
cloned identities together, and preserves known script, prompt, byte and PCM
connections. Unknown corpora with coincident numeric IDs are not assumed to be
the same person. New regressions cover these cases and the original failure.

Keep every one of the120 fixed confirmation foreground recordings. Retire the
four consumed originals connected to that fixed panel; do not replace the fresh
cases or select an easier subset after looking at model responses. Reassign the
remaining consumed connected components with the original deterministic role
policy. Tiny connected strata stay in development instead of being split.

| Role | Foreground originals | Canonical connected groups | Nine-condition inputs |
|---|---:|---:|---:|
| Fit |214|169|1926|
| Independent calibration |97|82|873|
| Consumed development |161|106|1449|
| Fixed confirmation |120|118|1080|

All66 background originals retain their separate roles.586 source roles remain
unchanged and retain all5274 exact original waveforms. Only six source roles
change; regenerate those54 inputs using the destination role's background pools
and the prespecified source/kind hash selection. All5328 logical cases remain
explicitly accounted for. Two genuine/cloned confirmation pairs now correctly
share their resampling groups;120 recordings are118 known foreground groups.

The metadata-only repaired plan SHA256 is
`f3d7a7f2e20a9d2aa9092c371f5af4d2a05ed9172e4652677049a8286f61605f`.
The full historical/identity audit SHA256 is
`2dafe3ca2021b40b298aa36d138d8ca481f0d7524a81cbb564749e8e3e72af7c`.
Its public digest-only receipt accompanies this record. Private manifests retain
original source text and local cache paths and are not published.

The [full historical/identity receipt](evaluation-panels/audio-v9-historical-identity-audit.json.gz)
and [corrected public source/recipe capsule](evaluation-panels/audio-v9-identity-plan.json.gz)
include only permitted source metadata and hashes. The latter's SHA256 is
`526a660d812c3afec959fb45946101d643181816e4b7c4248e93559280adbf83`.
The model-free canonical-component reviewer checked every role/recipe and the
original-input lineage; five deliberate actual-capsule alterations were refused.
All14 identity, repair and public-privacy regressions passed.
The selected source/calibration/guard/independent-review regression suite passed
all114 tests. The earlier frozen method's code and numerical requirements remain
unchanged.

The [corrected offline helper](../backend/scripts/rebuild_audio_identity_inputs_v9.py)
rebuilt all5328 inputs from the public capsule and separately acquired complete
original recordings, with zero failures. Its [complete receipts](evaluation-panels/audio-v9-public-reconstruction.json.gz)
have original JSONL SHA256
`eb44576b8d804126e6ea5c9069a9e9bf64b4cddea4c3bd7496db0814a65a7ec8`.
From the repository root, without a model or private manifest:

```sh
python3 backend/scripts/review_audio_identity_plan_v9.py \
  docs/evaluation-panels/audio-v9-identity-plan.json.gz \
  docs/evaluation-panels/audio-v8-source-plan.json.gz /tmp/audio-v9-review.json

PYTHONPATH=backend backend/.venv/bin/python backend/scripts/rebuild_audio_identity_inputs_v9.py \
  docs/evaluation-panels/audio-v9-identity-plan.json.gz \
  /path/to/separately-acquired-source-cache /path/to/new-output-directory \
  --capsule-sha256 526a660d812c3afec959fb45946101d643181816e4b7c4248e93559280adbf83
```

Originals retain their existing per-source terms and attribution. Source cache
entries are complete original bytes named`<sha256>.audio`. The original catalogue
and location supplement identify where to acquire them. Do not replace missing
or mismatching sources. These successful local reproductions do not establish
detector accuracy or participation by an outside reviewer.

## Requirements retained

Use the identical v8 frozen-feature learner, PCA16/ridges, independent guard
calibration, original raw-plus-calibrated stability checks, and actual guarded
service comparator. All16 advancement gates remain unchanged. The identity fix
changes source grouping and its denominators, not the acceptance/error policy.
Publish the corrected exact inputs and execution freeze before new extraction or
fitting. Exact-input features from the stopped run may be reused only after
weight/runtime/source/PCM/embedding parity checks; retain that lineage.

Complete source/recipe/native-forward/probe audits and a public numerical refit.
A development pass may permit a separately frozen full1080-input comparison;
neither source repair nor development alone enables a replacement. Paid-flow,
evidence/export and rollback checks remain required for any local adoption.
Image, video, writing and independent human evaluation remain outstanding parts
of the wider improvement program. GCP remains deferred, and no outside reviewer
participation is claimed.
