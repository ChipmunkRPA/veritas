# Noisy speech phrase localization on a separate test set

The comparison completed on 408 deliveries from 24 utterances and eight
readers. The data include 393 distinct audio byte sequences. Each utterance
provides two synthetic replacements, matched ordinary resampling controls, and
a clean original. The two noisy conditions add real traffic or separately
recorded competing speech that is 10 dB louder than the foreground at the
authored phrase. [Frozen protocol](audio-phase-partition-fresh-protocol-2026-11-18.md) · [Source dataset](https://zenodo.org/records/19522979)

An independent input review verified all sample clocks, all matched material
outside the labeled edit, and exact addition of the known background. Fifteen
repeated deliveries were identity resampling controls in the same source
group and condition. [Input custody receipt](evaluation-panels/audio-phase-partition-fresh-custody-2026-11-18.json)

## Fixed comparison

The candidate combines the existing two-second check with shorter windows
from an enhanced view. It requires two adjacent speech windows and stronger
response in the enhanced view than in the original mixture or any measured
remainder. The operating points came from earlier ordinary controls. Each
tested edit has a known source-sample interval, and a correct location
covers at least half of that interval with intersection over union of at
least 0.30.

| Condition | Existing correct locations | With the added check | Added ordinary alerts | Added wrong edit locations |
| --- | ---: | ---: | ---: | ---: |
| Clean | 15/48 | 26/48 | 0/72 | 0/48 |
| Traffic | 0/48 | 0/48 | 0/72 | 0/48 |
| Competing speech | 0/48 | 0/48 | 3/72 | 2/48 |

The clean gain covered six readers and both generator families. Three
competing-speech ordinary alerts came from one reader group. Among the 24
standalone backgrounds in each noisy condition, no file flagged. The
combined check had too little admitted speech to assess 21 edited traffic
deliveries and 33 ordinary traffic deliveries. The unchanged short-window
analysis of the original mixture found two competing-speech edits and
also flagged three ordinary files. The registered goal of additional
correct locations in competing speech was missed. The enhancement
comparison remains experimental with zero weight in the served assessment.
[Independent count replay](evaluation-panels/audio-phase-partition-fresh-decisions-2026-11-18.json.gz)

## Why the noisy cases were missed

The known intervals were inside the time grid. In traffic, only 2 of 48
edited deliveries had enough detected speech in two adjacent original
windows at the relevant location. The projected speech view had enough
activity in 34 of 48, along with 50 of 72 ordinary controls and zero of
24 standalone traffic recordings. This is a speech availability count.
The frozen method required the original activity gate before scoring the
projected view, so those additional detections of speech did not become
voice analyses.

With competing speech, all 48 edited deliveries had enough original
activity near the edit. Seventeen lacked relative support from the
projected view and the remainder. The other 31 failed the frozen response
point at that location. A separate readback of all 240 mixtures fitted the
two known audio sources to the exact enhancement used in the comparison.
At the edited phrase, median fitted gains for the selected view were 0.33
for the foreground and 0.055 for traffic. For competing speech, the gains
were 0.64 for the foreground and 0.79 for the interfering voice. These
diagnostics distinguish speech admission under traffic from foreground
separation under overlapping voices. They carry no authenticity finding.

## Replay the released counts

The release contains grouped decisions for each condition, generator role,
and comparison. Its separate reviewer rebuilds the reported counts and the
fixed decision criteria with the Python standard library. The independent
review matched each of the 408 delivered cases and all 2,040 decisions
against the private scoring ledger before grouping them. Media, exact time
positions, individual score vectors, and reader assignments stay in the
local study. The public replay covers counts and decisions. Recreating the
native signal measurements requires the frozen local inputs.
[Independent publication review](evaluation-panels/audio-phase-partition-fresh-publication-2026-11-18.json)

```sh
python3 -B -I -S backend/scripts/review_audio_phase_partition_decisions.py \
  docs/evaluation-panels/audio-phase-partition-fresh-decisions-2026-11-18.json.gz \
  docs/evaluation-panels/audio-phase-partition-diagnostic-2026-11-18.json.gz
```

## Follow-up on traffic speech admission

A separate retrospective run scored 1,100 additional windows in the 144
traffic deliveries. It admitted a window when the projected view had enough
speech and neither the original nor the remainder met the original speech
requirement. It kept the existing response point and neighbor rule. The
independent count confirmed zero correct locations out of 48 edited files,
three flags out of 72 ordinary files from one reader group, and zero flags
on the 24 standalone traffic deliveries. More measurable speech did not
improve these localization results. The follow-up reused the diagnosed
recordings and carries no claim of independent accuracy.

## Agreement at identical window positions

A retrospective readback required the same two neighboring short windows
to clear their existing points in both the original and projected views.
Adding this rule to the two-second check located 24 of 48 clean edits
with zero ordinary alerts, up from 15 of 48. Adding the raw short-window
rule instead located 29 of 48 clean edits. Under competing speech the
raw addition located two edits, flagged three ordinary files, and gave
two misplaced alerts. Agreement between views gave zero alerts and zero
correct locations under competing speech or traffic. Both views use the
same detector and the same recorded signal. The result motivates
a separate evaluation of agreement before any product change.

## Fixed score corrections on the original speech windows

Two follow-ups replayed all 408 deliveries and 9,384 source-window admissions
without running the voice detector again. A scalar correction fit separately
on MLS foreground speech was evaluated on two other generator families. It
was tested at its frozen wider point. A second comparison kept that point
and required both short-window responses to exceed assessed non-overlapping
speech on either side by at least 1.5 or 3.0 raw response units. Each arm
kept the current short-window and two-second findings. A separate arithmetic
recount verified the scalar cutoff and exact time overlaps.

| Condition | Measurement | Current | Wider alone | Wider with 1.5-unit contrast | Wider with 3-unit contrast |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean | Located edits / 48 | 29 | 36 | 36 | 36 |
| Clean | Ordinary alerts / 72 | 0 | 3 | 3 | 3 |
| Traffic | Located edits / 48 | 0 | 0 | 0 | 0 |
| Traffic | Ordinary alerts / 72 | 0 | 0 | 0 | 0 |
| Competing speech | Located edits / 48 | 2 | 5 | 4 | 2 |
| Competing speech | Ordinary alerts / 72 | 3 | 30 | 30 | 9 |

All arms left the isolated backgrounds unflagged. With the wider point
alone, one of three newly located competing-speech edits cleared every
matched ordinary clock. The contrast additions supplied zero such noisy
locations. Only 3 of 48 edited traffic recordings admitted an eligible
short-window pair at the edit clock. With competing speech all 48 did, but
the short-window responses ranked the edited signal higher than its matched
genuine control at only 24 of 48 edit clocks. That comparison was 48 of 48
for clean speech. The foregrounds and generator families were separate
from fitting, while all four background captures had been used in the
fitting or calibration panel. The frozen correction remains outside the
app. These results distinguish insufficient speech admission from weak
discrimination when two voices overlap.

The brief speech evidence view and PDF now distinguish checked time, time
with limited or uncertain speech, and time outside the check's schedule.
For the supported twelve-second recordings, the original sample positions
and all 23 window admissions appear alongside the selected listening ranges.
The final 250 ms is outside this schedule. The display follows the retained
admission data from each analysis. Six previously paid local analyses
reproduced their signed original-file evidence and the same JSON exports
when rendered with the new report layout.
