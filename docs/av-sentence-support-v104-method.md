# Sentence-connected audiovisual roles and native time support

This metadata preparation addresses two concrete issues in the
[qualified audiovisual panel](video-native-source-v92-result.md): sentence
contexts crossing roles and an audio origin preceding the first picture.
It creates a new study view of all 48 existing sources. No old source plan,
role, frame record, source media or experimental result is modified. It is
not a new decoder pass, generator qualification or detector comparison.

## Connected roles

Parse the publisher filename's actor and sentence fields, requiring agreement
with the locked source group. Join sources sharing a sentence, actor or known
ancestry component. Assign the complete connected component to one study role.
Read only the 33 historical manifests already committed to the original v89
source plan, checking their exact hashes. Any component containing a sentence
observed in those histories is fitting-only, including its linked sources.

Order the remaining components by a fixed hash salt. Search all assignments to
confirmation, confidence-calibration and selection with at least two components
per role. The bounded design permits six to ten remaining components. Minimize
source-count range, then squared source counts, then component-count range;
break ties by the assignment tuple in that fixed order. No detector score,
truth outcome or visual inspection chooses a role. This balances the small
panel without splitting shared sentences across roles. Fitting uses the known
historical contexts. These are new study roles; prior roles remain recorded.

Only 12 sentence contexts are available. Actor codes and recorded ancestry
are not independent proof of distinct people, absence of near duplicates,
complete historical audio separation or unknown pretraining independence.
Calibration or confirmation on only a few sentence components cannot establish
population performance or authorize a new customer confidence contribution.

## Preserve both stream clocks

Use the recorded integer picture PTS and rational time base, refusing truncated,
interpolated or unordered clocks. Require qualified contiguous audio-frame
records. Verify each audio interval against its native timestamp, sample count
and sample rate. Retain the native audio-frame timestamp quantization; permit
only the already qualified one clock tick plus one sample deviation between
frames and from the cumulative sample lattice. Record that allowance and the
largest observed cumulative deviation explicitly.

Record source audio and video origins separately and retain their signed
difference. For every common picture center, record its exact declared source
time, time relative to audio origin and all audio-frame intervals containing it.
Small quantization gaps or overlaps remain visible as missing/ambiguous mappings.
Also record a uniform native-sample lattice conditional on contiguous decoded
sample order. This preparation has not independently decoded native PCM or
qualified a resampler's phase. It must not declare sample-exact conditioning or
erase the original offset by starting both streams at zero.

Common support ends conservatively at the last observed picture center within
both declared audio support and the conditional cumulative sample extent. No
display duration is invented from a nominal frame rate. Preserve unsupported
sources without padding or repeating them.

For subsequent visual construction planning, a native 20-frame edit requires
at least 24 common observed pictures, including two context pictures on either
side. Center that interval and retain actual before/during/after frame indices
and source times. Twenty native pictures are not necessarily 20 pictures at
25 Hz. This is a geometry plan, not a MuseTalk/other generator readiness test;
`audio_conditioning_ready` remains false for every source. Generator-specific
timing, native PCM/sample-count verification and achieved phonetic alignment
remain separate work.

## Execution and publication

Freeze this source-only implementation, authored tests, procedure, original
inventory/qualification/asset-plan hashes and historical bindings before building
the new private plan. Manually publish and verify the freeze first. Use exclusive
output files/directories and preserve failed attempts. No automatic retry,
overwrite or previous-role mutation. The single-process metadata stage uses
900 seconds, 1.5 GiB self RSS, 3 GiB initial RAM/disk and 1.5 GiB running reserves,
observed per source rather than continuously. Existing neural limits remain.

Publish only aggregate counts, code and whole-artifact commitments. Source names,
actor/sentence pairings, per-picture/audio mappings and input fingerprints remain
private. Existing publisher terms and attribution limits remain those of the
[original source preparation](video-common-source-preparation-v89.md). No raw
media acquisition, neural forward, customer job, outreach, Actions or GCP action.
