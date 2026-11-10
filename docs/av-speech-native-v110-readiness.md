# Native speech encoder: software ready, neural qualification pending

The [frozen contract](av-speech-native-v110-method.md) now has a CPU encoder
adapter, separate layer-state captures, native-clock routing, deterministic
replay checks and the predeclared suffix-context probe. Its three cached
Apache-2.0 speech assets match their publisher-bound records. The freeze was
pushed in `e5230a9` before the new process preflight.

All **49 local software tests passed**: 19 new PCM/state-packing/runtime tests
and 30 retained clock-routing tests. They use authored arrays and mocked
resource observations. No pretrained encoder, actual hidden-state sequence or
generated media was tested by these unit tests.

The new owned-process preflight exited zero, with no guard or sampling error.
It enumerated and sampled a live standard-library child/grandchild and verified
exit. Its internal live-process RSS observation was **28,049,408 bytes**.
The complete monitored stage took **0.263 seconds**; observations are not
guaranteed peaks. The shared helper's original v108 receipt schema is retained
explicitly in the [readiness receipt](evaluation-panels/av-speech-native-readiness-v110.json).
This was a distinct successful preflight and did not retry the failed historical
audio extraction.

The subsequent read-only neural admission check reported **5,765,709,824 bytes
available**, below the required **6,442,450,944 bytes (6 GiB)**. It created no
attempted authored stage and made no pretrained forward. No resource limit,
evaluation point or model calibration was changed to bypass admission.

The actual next steps remain the three authored forwards, native feature/source
qualification, mouth generation with achieved alignment checks, and a fresh
articulation comparison. The completed [video input bank](video-native-inputs-complete-v108.md)
also awaits its native neural comparison. These results establish software and
process readiness only; new detection accuracy and customer confidence remain
unqualified. Source features, raw voices and model weights were not published.

Serving behavior, old model rollback, billing and report limitations are
unchanged. No customer analysis, outreach, workflow, real payment or GCP action
occurred.
