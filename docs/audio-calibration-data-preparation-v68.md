# Separate source preparation for combined audio confidence

The [complete v64 comparison](audio-shared-confirmation-v64-result.md) recovered
91 correct conclusions with the pair, but added 28 wrong original tendencies.
Twenty-five of those extra errors were in its expressive ElevenLabs stratum.
Keep the useful combination and test its calibration on separately assigned
sources, rather than treating standalone rejection or any extra error as final.
This release prepares candidates; it does not fit or enable a confidence model.

## Verified preparation

| Source bank | Actual work | Remaining qualification |
| --- | --- | --- |
| Archived expressive synthesis | All 17 original archives, 991,109,120 bytes, verify against recorded publisher SHA-256. All 432 English generation sidecars match the publisher index. There are 408 candidates with previously unrecorded bytes and declared scripts. | Verify selected payload hashes and production PCM. All 24 voice profiles appeared historically, so these are new recordings/scripts with seen voices, not entirely unseen voices. |
| Genuine speech | Four original LibriSpeech `train.360` Parquet files, 1,950,348,042 bytes, downloaded to the authorized external development drive and verify against pinned publisher SHA-256. Their metadata contains 8,668 recordings, 77 voices and 59 voices absent from the recorded history. All declared scripts are absent from that history. | Select and decode complete original recordings; verify source/PCM/text/role connections. No audio from this bank has been sent through a model. |
| Archived cloned speech | Four existing publisher-verified PLN archives, 6,031,923,200 bytes, contain 3,452 user utterances. All WAV headers inspected successfully. There are 690 at-least-four-second candidates without known archived-member or declared-text reuse, covering 295 voices; 197 voices are absent from recorded history. | Verify selected full payload hashes and production PCM. Parent-speaker mapping is publisher provenance, not biometric identity verification. Only the publisher-declared MIT PLN scenarios are included. |

The cloned-voice bank has 184 history-unseen parent identities after excluding
every genuine-bank identity. Of those, 107 have at least two distinct qualifying
scripts, 59 at least three and 29 at least four. The genuine and cloned namespaces
are joined through their declared LibriSpeech parent; they are not treated as
unrelated identities merely because one file is synthetic.

The stock-generator companion metadata retains the already qualified
[Kokoro runtime](audio-generator-qualification-v57-result.md). Its exact pinned
publisher revision has 28 English profiles; 16 are absent from recorded history.
Their declared American/British profile groups and any new auxiliary assets still
need a source-generation freeze before new generation. The new corpus will not
be described as containing an entirely new generator architecture.

## Independence and calibration design

Global recording, script and voice novelty are different properties. A repeated
stock voice does not make new audio identical to old audio, but historical voice
reuse limits an unseen-speaker claim. Report these properties separately. Do not
discard every useful provider simply because its finite stock-voice bank appeared
in an earlier study, and do not silently relabel that bank as unseen voices.

Assign fitting, confidence-calibration, method-selection and confirmation sources
before model calls. Keep known voices, cloned/genuine parents, identical declared
text/prefixes and actual byte/PCM connections within their assigned roles. Retain
a separate unseen-voice stratum and a seen-voice/new-script expressive stratum.
Matched genuine-script synthesis can help distinguish synthesis effects from
literary-versus-conversational content differences; its paired sources must share
the same role and resampling group.

Source eligibility may use provenance, duration, channel/sample-rate headers,
known identity connections and production-decoded integrity. Do not choose cases
from upstream humanness ratings, ASR review recommendations or detector scores.
All such annotation fields are ignored. Current inventories retain short, reused
and failed candidates explicitly; no final study roles or final source panel have
been assigned yet. Full decoded-PCM novelty is not inferred from metadata.

The next experiment should fit the combined confidence using only its fitting
role, calibrate on its independent calibration role, compare on selection and
retain final confirmation separately. Judge useful gains, both wrong directions,
unknown results, calibration and background conditions together. Agreement or an
average of component confidence is not a combined calibrated probability. No
public screening threshold or accuracy guarantee is introduced.

## Permissions and verification

Genuine speech retains the publisher's CC BY 4.0 terms and pinned revision
`71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`. Expressive synthesis retains the
recorded CC BY 4.0 release terms and model/source provenance. Cloned speech uses
only publisher-declared MIT PLN scenarios with their original LibriSpeech-parent
attribution. These declarations do not establish complete depicted-person consent,
biometric identity or absence from pretrained model data.

All 11 focused source-provenance, novelty, permission, size and annotation-exclusion
tests passed, as did lint. No generator, detector, VAD or learned calibration was
run. WAV-header inspection is not production PCM qualification or model accuracy.

See the [expressive inventory](evaluation-panels/audio-expressive-inventory-v66.json),
[whole genuine-file receipt](evaluation-panels/audio-genuine-sources-v67.json) and
[joint candidate summary](evaluation-panels/audio-source-inventory-v68.json).
Only code, aggregate metadata and whole-file hashes are published. New recordings,
individual source identities, text, voice features, pairings and learned artifacts
remain private. No customer upload, charge, workflow, outreach or GCP action.
