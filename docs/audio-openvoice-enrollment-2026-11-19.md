# Learned extraction with a second voice-conversion pipeline

On four held target readers, adding the enrollment-conditioned processed view gained **zero correctly
located noisy edits** and added **17 genuine-recording alerts across three reader
lineages**. Processing alone lost both held clean detections. The acoustic and
paired-detection requirements both failed. The extraction candidate remains experimental.

## Inputs and source scope

Six targets were fixed before model responses, with two development and four held.
Each target used four different genuine utterances for a twelve-second host,
four-second detector enrollment, separate four-second conversion conditioning and
an ordinary splice donor. The two reference clips came from different chapters.
These targets differ from the preceding learned-extraction target set. Their
reader identities and source corpus have historical project exposure.

The new conversion pipeline applies the official MIT OpenVoice V2 checkpoint to
two retained MeloTTS-rendered utterances from the CC BY 4.0 PhonemeDF release, one
base utterance per role. Six target-conditioned outputs qualified using frozen
native waveform checks. This adds a conversion architecture. It supplies neither
an independent source corpus nor biometric verification of speaker identity. The
historical Melo checkpoint is unverified. The official Melo model terms were
audited separately, and no new Melo checkpoint was loaded.

Two retained AMI meetings provide development and held backgrounds. Their
participants are disjoint across roles and the preceding trial's meetings. One
held meeting is shared by all four held targets. **Physical room independence is
unestablished.** The source meetings have prior project exposure. Retained native
bytes, original clocks and annotated overlapping-speech ranges were independently
replayed. An optional new publisher range request returned HTTP 503. The retained
sources were preserved unchanged.

At the same fixed 960 ms native interval, the panel includes synthetic conversion,
untouched genuine speech, same-reader genuine splice, different-reader genuine
splice and ordinary local resampling. Each is delivered clean and with competing
speech at nominal −5 and −10 dB. The variants share fades, peak-safe gain, background
and every sample outside the edit. There are **102 deliveries: 18 synthetic from
six physical edit events, 72 genuine and 12 scaled isolated backgrounds**. Repeated
conditions and background copies retain their source-group relationships.

## Frozen comparison and held results

The exact prior learned-extractor runtime produced 120 raw responses: true
enrollment on all 102 inputs and wrong-enrollment diagnostics on 18 synthetic
inputs. A deterministic repeat gives 121 calls in 372.89 seconds. The genuine
component passed the retention guard in 12/12 held untouched cells. The synthetic
component passed in 0/4 clean, 3/4 −5 dB and 4/4 −10 dB cells. Only 7/8 noisy
untouched cells met the SNR-improvement guard. These measurements concern the
same-clock waveform component and establish no authorship or speaker attribution.

The forensic comparison ran the actual current 750 ms check on 222 views. Its
23-window grid, 500 ms step, speech admission, strict raw point
`6.642364579351768` and consecutive-window rule stayed frozen. The last 250 ms is
uncovered. A useful selected interval must cover at least half the known edit and
have temporal IoU at least 0.30. The additive arm preserves every original finding.
Wrong-enrollment diagnostics never enter that arm.

Each forensic view is analyzed as one file. A separate genuine utterance supplies
enrollment for processing. Immutable host, replacement and background components
support offline native-clock grading only. They are never detector inputs. This
trial makes no comparison against an immutable earlier copy at inference. Native
source and runtime pins were checked before inference and again at final review.
The isolated background controls remain separate from every genuine mixture.

| Held condition | Useful original | Useful processed | Useful additive | Genuine alerts original → additive |
|---|---:|---:|---:|---:|
| Clean | 2/4 | 0/4 | 2/4 | 0/16 → 0/16 |
| −5 dB competing speech | 0/4 | 0/4 | 0/4 | 4/16 → 12/16 |
| −10 dB competing speech | 0/4 | 0/4 | 0/4 | 1/16 → 10/16 |

The additive arm leaves 10/12 held synthetic deliveries without a useful
localization, including all eight noisy deliveries. At least some speech was
assessed in every file. The edit itself remained unassessed in one held synthetic
and two held genuine deliveries. Those gaps are included in the no-useful-result
counts and never establish originality.

Held synthetic alerts confined to the wrong time rose from 1 to 3 at −5 dB and
from 0 to 2 at −10 dB. Genuine highlighted time grew from 6.25 to 58.00 seconds.
Held genuine alerts by control type, each across 12 condition deliveries, were
untouched 1→5, same-reader splice 1→6, different-reader splice 2→6 and local
resampling 1→5. Isolated held backgrounds stayed at 0/8 alerts. Across development
and held roles together, 21 new genuine files across four reader lineages were
flagged, and processing alone lost three original useful detections. Neither role
gained a correctly located synthetic edit.

The forensic run took 872.56 seconds, with 2,489 distinct classifier inputs and
one exact repeat. Independent review checked all 222 native views, 42,624,000
sample values, 5,106 scheduled windows, 4,441 admitted-window bindings and all
108 decision groups. Earlier independent checks replayed 19,584,000 input samples,
12,718,080 outside-edit identities and 23,040,000 processed-output values.

## Recount, rights and reuse limits

The accompanying numeric receipt contains grouped decisions, paired transition
counts and anonymous event-count histograms. The following recount checks
the pinned bytes, denominators, gains, losses, reader counts and frozen gates
using the Python standard library. [Numeric receipt](evaluation-panels/audio-openvoice-enrollment-2026-11-19.json)

```sh
python3 -B -I -S backend/scripts/review_audio_openvoice_enrollment.py \
  docs/evaluation-panels/audio-openvoice-enrollment-2026-11-19.json
```

A separate privacy/tamper review rejected 15 authored corruptions and excluded
1,255 exact private source tokens.

The public package omits raw recordings, generated speech, voice embeddings,
source identities, recording hashes, file paths, per-case offsets and classifier
response vectors. Exact waveform, VAD and model replay require the private custody
package and licensed source material. The portable recount validates the released
aggregate arithmetic and paired counts.

The [OpenVoice source](https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/LICENSE)
and [exact converter model card](https://huggingface.co/myshell-ai/OpenVoiceV2/blob/f36e7edfe1684461a8343844af60babc2efbb727/README.md)
state MIT terms. The [Melo model card](https://huggingface.co/myshell-ai/MeloTTS-English/blob/bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef/README.md)
was audited separately. Source data retain [PhonemeDF CC BY 4.0](https://zenodo.org/records/19522979),
[LibriSpeech CC BY 4.0](https://www.openslr.org/12/) and
[AMI CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) terms. The
[prior extraction study](audio-learned-enrollment-genuine-preflight-2026-11-19.md)
documents the reviewed extraction checkpoint and upstream components.

LibriSpeech train-clean-100 is also named by the extractor's published training
recipe, so training overlap is possible. Six target readers, two reused base
utterances and one held background meeting limit generalization. This engineering
result supports keeping extraction out of the proposed detection path. It supplies
no commercial accuracy estimate, confidence calibration or court-evidence claim.
