# Short speech and mouth comparisons on native 25 fps AVC

The optional development Full route reads one native mono 44.1 kHz AAC-LC MP4 soundtrack and the original 360 × 288 AVC1 4:2:0 pictures. It requires 75 verified pictures at exactly 25 fps on the zero-based 1/90,000 source clock and 47,648 presented 16 kHz samples. The existing speech and mouth timing and 44.1 kHz motion flags must be enabled before `VERITAS_VIDEO_LIP_AAC441_NATIVE25_REVIEW=true` can select this route. The new option defaults off. Existing near-silent soundtrack results keep priority wherever eligible.

The added shot check compares adjacent image changes against the nearby frame transitions. It requires evidence that the change persists when comparing three frames on each side, and withholds the supplemental decision where the luminance structure still aligns after a small translation. The existing shot call and its boundaries remain intact. A reported change withholds the speech and mouth comparison. The cut carries zero AI-authorship weight and can occur in ordinary editing. The browser, PDF, and JSON record the same native position. Signed replay with the original recomputes the shot check and verifies the mono soundtrack clock. If the speech comparison reports a range, the packet also has its exact supporting pictures, mouth crops, waveform, and audio excerpt.

## Independent held cut check

All 28 newly selected exact recordings came from 14 retained, documented CC BY 4.0 GRID publisher archives. The two recordings and speakers used to discover the prior cut failure were outside the holdout. Other speaker groups had prior project exposure. The sequence of source selects, construction, cutoff, and count criteria were fixed before the new measurements. The cuts join genuine recorded pictures at frame 37, which is 1.480 seconds or PTS 133,200 on the 1/90,000 clock. The test used controlled joins in short speech recordings. Rates on natural movie edits, multiple people in the same recording, and unrelated camera styles remain unmeasured. [GRID publisher](https://zenodo.org/records/3625687)

| Held case | Existing shot check | Reduced cutoffs alone | Existing plus guarded change |
| --- | ---: | ---: | ---: |
| Across-speaker cuts, exact | 6/14 | 12/14 | 14/14 |
| Same-speaker cross-take cuts, exact | 3/14 | 6/14 | 7/14 |
| Same-take jump, exact | 0/14 | 0/14 | 0/14 |
| Wrong cut locations | 0/42 | 0/42 | 0/42 |
| Unchanged or continuous stress control exclusions | 23/126 | 42/126 | 23/126 |
| Untouched exclusions | 0/28 | 0/28 | 0/28 |

The additive check located 12 additional cross-take cuts across ten held source groups. All 23 ordinary exclusions with that change enabled came from the incumbent method on authored flashes or saturation changes. The additional check produced no ordinary exclusion, including against strong same-recording exposure, focus, translation, drift, saturation, flash, and smear controls. Same-recording jumps remain a known miss. The prior discovery cut replayed at its exact boundary. The decision is an eligibility control. An abrupt boundary can be ordinary editing.

The native comparison recorded 12,432 frame slots over 168 cases. A separate MP4 replay verified 12 added cuts and three pre-selected untouched controls. The decoded RGB, exact clocks, AAC origin, and 36 cut-witness frame anchors agreed. Core scan took 24.7 seconds, and the physical MP4 replay took 15.9 seconds.

## Separate soundtrack carrier comparison

The existing frozen speech comparison localized 2/16 staged speech replacements on both 44.1 and 48 kHz. Both carriers misplaced one edit, flagged one of 64 ordinary examples, one of 16 matched same-word dubs, and one of 32 whole-track delays. Each selected interval and all three supporting source frames agreed across the two carriers. These are paired comparison counts from a separate cohort. The results describe staged speech replacements within that carrier profile. The new cut check addresses the admission gap, and this paired reading supplies a narrowly validated physical timing observation for sources the guard admits.

## Local app verification

Eight simulated USD 9.99 checkouts queued their analyses after payment.
Browser, PDF and JSON evidence agreed for all six files with the option
enabled. Two cuts withheld the comparison at source frame 37
(1.480 seconds). Two ordinary clips had no qualifying mismatch. Two
staged speech replacements showed original frames 37, 40 and 48 at
1.520–1.920 seconds and frames 45, 48 and 52 at 1.880–2.120 seconds.
The signed packages reproduced the original picture and sound evidence.
Each of two paired option-off analyses had the same overall assessment
as its option-on replay. The local run kept the older near-silence option
enabled.

## Within-recording omissions

Ten separate GRID source groups supplied 30 authored omissions with the
picture and sound edited together, plus 70 matched controls per carrier.
The existing cut check located zero of the 30. A separate trial measured
small foreground motion after accounting for camera translation and
compared it with abrupt changes in the soundtrack. With the operating
point fixed on development groups, it located 6 of 30 AAC and 7 of 30 PCM
omissions. It placed zero findings at wrong times and flagged three AAC
and two PCM controls. The test observed no flags on untouched clips,
smooth retiming, lighting changes, camera pans, or gain controls.
[GRID source collection](https://zenodo.org/records/3625687)

Native MP4 replay verified all six correctly located AAC cases and three
flagged control cases. It reproduced 567 decoded frames, 27 native frame
positions and the original audio samples. After the holdout closed, an
exact-timestamp requirement kept all correct locations and reduced the
alerts to one picture-only omission per carrier. This follow-up used
already measured clips. The method remains experimental. An abrupt
timing change can occur in ordinary editing and carries no inference
about who or what produced the file.

A second test fixed the exact-timestamp rule before running it on 14
fresh native source recordings, including six from two previously unseen
participants. In clear audio it located 5 of 42 AAC omissions and 4 of 42
PCM omissions, with zero misplaced findings and one picture-only edit
flagged among 98 controls per carrier. Under strong background noise
(−5 dB), the rule located none of the 42 omissions under either background
condition. The picture cue still passed at 29 of those boundaries, while
the soundtrack cue weakened. Two separate recordings adapted from 30 fps
also produced too many ordinary alerts to support broader use.

Original-file replay checked 27 MP4s, all 1,701 decoded picture timestamps
and frames, 60 before, at and after anchors, and exact decoded audio. The
rule remains experimental.
