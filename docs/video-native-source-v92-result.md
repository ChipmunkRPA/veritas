# Shared video sources: completed native qualification

All **336 acquired assets** decoded successfully under the
[published procedure](video-native-source-qualification-v92.md). The audit read
**166,569 native RGB frames** and their presentation timestamps. All 336 video
clocks passed; all 48 available audio streams passed frame/sample-clock and
quantization-tolerant continuity checks. No detector, generator or customer file
was processed. No serving default changed.

The [aggregate receipt](evaluation-panels/video-native-source-qualification-v92.json)
binds the private qualified source plan, per-batch resource profiles and separate
recorded-clock reviews. Sources and individual fingerprints remain private.
Code/freeze publication preceded decoding; all verification ran locally and
GitHub Actions stayed disabled.

## What can be tested

| Sources | Assets | Audio available | Suitable role |
| --- | ---: | ---: | --- |
| Paired original footage | 96 | 0 | Visual genuine controls |
| First swap method | 96 | 0 | Visual face-swap checks |
| Second swap method | 96 | 0 | Visual face-swap checks |
| Recorded audiovisual speech | 48 | 48 | Speech/mouth controls and separately prepared manipulations |

The 288 silent assets cannot establish lip-sync performance. Audio presence in
the other 48 is a technical availability measurement, not proof of intelligible
speech, correct speaker attribution or achieved synchronization.

Exact informative RGB/PCM comparisons found no additional cross-group links
within this panel, and no matches to the recorded historical RGB sequences.
The 144 known source components therefore retain their provisional ancestry
roles, 36 groups per role. This does not prove absence of near duplicates,
re-encodes, shared people, unrecorded historical audio or pretrained overlap.
The publisher's missing portrait attribution/identity links remain unresolved.

## Additional preparation issues found

A separate rational-clock review recomputed all picture timestamps and 48 audio
sample-clock records from the saved journal. Canonical audio sample counts differ
from the exact native-rate conversion by at most **0.654 of a 16-kHz sample**,
consistent with rounding. This recorded-data review is not a new decoder pass,
truth adjudication or outside human review.

Only **90/96 swap groups** have exactly shared native timestamps and dimensions
across the original and both swaps. The remaining six need explicit source
correspondence before controlled insertion, or must remain unavailable for
that construction. Equal frame indices cannot be assumed to represent the same
time. All six remain in the source inventory; no source was replaced based on
detector responses.

The 48 speech sources cover 12 stock sentence types. **Eleven sentence types
cross the current roles**, and four types appear in recorded prior experiments.
These are known shared language contexts, distinct from physical-file ancestry.
A learned lip comparison needs sentence-aware roles and clear historical-context
limits before fitting. Different actors alone do not establish unseen utterances.

Conservative common picture-center/audio support ranges from **0.533 to 4.471
seconds**. One source is under 0.8 seconds; 24 are under 2.5 seconds and 47 under
four seconds. This intersection ends at the last observed picture center without
inventing its display duration. The shortest source cannot support the usual
20-frame, 25-Hz mouth edit with surrounding context. Retain it as unsupported
where necessary, without padding, repeating content or declaring it original.
A four-second-only audio calibration cannot be applied automatically to these
shorter clips.

Audio starts 25–27 ms before the first video frame in these recorded files.
Preserve both origins when constructing aligned crops; copying PCM from sample
zero into a newly zero-origin picture clip would change their relationship.
Clock availability/continuity alone does not verify phonetic synchronization.

## Execution and remaining work

All twelve decoder stages and finalization passed, with zero guards or sampling
errors. Decoder-stage time totaled **1,884.81 seconds**; maximum sampled
descendant RSS was **604,143,616 bytes**, and minimum observed available RAM was
**5,098,389,504 bytes**. These sampled observations are not guaranteed peaks.
Finalization also terminated successfully. Verified descendant exits were
recorded separately from sampling failures.

The previously reviewed [adapted mouth generator](mouth-multifamily-v7-evaluation.md)'s seven cached files were
also rehashed against its original publisher-bound plan: **3,885,968,140 bytes**
match the retained acquisition receipt, with zero downloads or neural calls.
That read-only cache check does not establish new generated-clip quality.

Next, prepare exact insert boundaries, before/during/after support, ordinary
controls and new mouth edits under a fixed source-correspondence and sentence
role scheme. Then compare the current visual, face-appearance and mouth/audio
methods on shared applicable inputs with their own fixed calibration/selection
roles. This source qualification is complete; **detector comparison and serving
qualification remain incomplete**. Source clocks and metadata are not deepfake
accuracy, pixel authorship or court-reliability evidence. No outreach, real
payment or GCP action occurred.
