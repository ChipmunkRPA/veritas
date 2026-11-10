# Native mouth clips: first construction snapshot

The first three of eight frozen six-source batches completed successfully:
18 source groups, 108 planned condition records, **103 ready complete clips**
and **five unavailable edits** for the short source. Its original clip is retained.
All three stages exited zero without resource guards or sampling errors. They
made 3,140 neural inputs in total, including explicit seeded repeats. The
[partial execution record](evaluation-panels/native-mouth-clips-progress-v120.json)
contains actual receipt/profile hashes and resource measurements.

Every ready clip passed the constructor's full decoded RGB, native picture
clock, soundtrack sample and audio-origin comparisons through the qualified
[assembly path](native-av-mux-v119-result.md). Exact edited-frame masks and
before/during/after frame positions are saved privately. Clips, source pairings,
face/voice states and per-source fingerprints stay on Seagate.

The same guarded controller continues the five remaining batches sequentially,
without retry or overwrite. This snapshot covers only the first three completed
batches and remains historical after future completion. The separate
[source-to-mask audit](native-mouth-clip-review-v121-method.md) is frozen but has
not executed at this snapshot. It must recompute actual pixel differences and
check masks/positions against complete source and output decodes.

No detector has scored these clips. Known generation provenance and correct
construction do not establish achieved phonetic synchronization, naturalness,+detection accuracy or a reason to promote a model. New articulation comparison,
source-held-out errors, calibration and full customer-flow checks remain required.
Serving model choices, confidence and rollback are unchanged. No external contact,
Actions workflow, real charge or GCP deployment occurred.
