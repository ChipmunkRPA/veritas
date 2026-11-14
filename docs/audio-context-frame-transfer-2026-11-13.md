# Comparing the same speech interval in overlapping model windows

An unusually high two-second score can come from the background or from
one part of the window. This test checks whether the same 800 ms source
interval produces an elevated response when the classifier sees it in
more than one surrounding context. It uses the original recording and
the installed speech model. Each reported interval is a timing lead for
human review. [Prior random-clock noisy panel](audio-random-clock-stock-2026-11-13.md)

## Method

On the existing five two-second windows, capture the unchanged head’s
response at every valid encoder step. The frame means must reproduce
the existing whole-window result. The 25 ms acoustic frames have a 10 ms
hop and are stacked in pairs for the encoder, yielding response centers
at sample 280 and every 320 samples thereafter in each window. Align
these to the original 16 kHz clock and average the responses in 400 ms
cells with at least 12 frame centers. Each candidate interval has two
adjacent cells, and each cell must have responses from at least two
model windows. Use the weakest observed response over both cells.
Since windows overlap and their encoder receives surrounding speech,
this is sensitivity to context within the same recording. It is a
correlated cue with limited coverage at the start and end of the clip.
[Previous temporal head readout](audio-temporal-head-transfer-2026-11-11.md)
· [Research on local temporal variation](https://arxiv.org/abs/2507.15101)

Compare the shared response by itself, its excess over distant parts
of the same recording, and a rise followed by a return. The distant
arm adds the difference between the shared response and the median of
available cells beyond 800 ms on either side. The rise-and-return arm
requires the shared response to exceed the mean response before and
after the interval, and adds twice the difference from the higher
flank. Compare lowering the current whole-window point without any
frame logic. Existing flags are preserved throughout.

For each arm freeze three points on the 44 already scored severe-noise
and isolated deliveries from the first stock-voice test. The points
allow zero, one, or two additional genuine reader groups from that
panel and zero additional isolated backgrounds. Then run 43 inputs
from the separate randomized-clock panel and 14 timing-aligned
same-word inputs from PhonemeDF. The first transfer contains 16
synthetic deliveries from eight host groups, 24 matched ordinary
controls and three isolated backgrounds. The second has four
replacements, eight real or ordinary-edit controls and two isolated
backgrounds from two held source readers. Waveform hashes match the
prior native receipts. These are new readouts of historical panels.
The same-word set uses exact publisher text alignment and two
synthesis mechanisms, with one historically exposed reader.
[Stock-voice source material](audio-stock-voice-transfer-2026-11-13.md)
· [Commercial model and source terms](commercial-model-audit.md)
· [Same-word source panel](audio-phoneme-short-edit-intake-2026-11-13.md)

An indicated range is useful when it covers at least half the known
edit and has temporal IoU of at least 0.30. False alarms, misplaced
flags, and admitted windows without flags are counted independently.
The ordinary alarm budget counts a reader once across its matched
controls. No customer threshold or confidence is fitted from these
readouts.

## Stock-voice transfer

| Route at its most permissive frozen point | Useful locations | Wrong-window flags | False ordinary deliveries | False ordinary groups |
| --- | ---: | ---: | ---: | ---: |
| Existing short review | 1/16 | 1 | 3/24 | 2 |
| Lowered whole-window point | 2/16 | 1 | 5/24 | 3 |
| Same interval across two contexts | 1/16 | 1 | 4/24 | 2 |
| Same interval above distant context | 1/16 | 1 | 3/24 | 2 |
| Same interval above both flanks | 1/16 | 2 | 4/24 | 2 |

At all three frozen points, none of the aligned-frame rules recovered a
correctly placed stock edit. The strongest raw window aligned usefully
on nine of 16 edits before applying a point. The absolute and distant
context rules each placed four of 16. The two-sided return could choose
an interval on six edits and placed two usefully. The control comparison
shows that changing the score threshold and changing interval placement
solve different problems. Lowering the whole-window point only recovered
one additional edit. Its **4.000–6.000 s** range covered the known
**3.941–4.901 s** insertion, with two more ordinary false flags.

## Same-word transfer

The existing short review and every candidate located zero of four
same-word replacements at −5 dB. Before choosing a point, the strongest
raw window aligned usefully on three replacements. None of the new
rules produced a useful best range. The two-sided return picked one
misplaced range at the more permissive points. Every route left all
eight matched ordinary controls and both isolated backgrounds below
its point.

On the fitting stock panel, distant context found one extra Piper
splice at the point allowing one additional ordinary group. The
surrounding-context minimum found that same splice when allowing two.
The advantage did not repeat on the random-clock or same-word panels.
An earlier whole-window-only replay likewise found no added correct
random-clock location at points allowing zero or one additional fitting
reader. These tests suggest that a fixed clip-level head’s strongest
local response is often reacting to some other part of a strongly mixed
recording. They leave a separate question for future training on
explicit timing and target-speaker labels. This test supports keeping
the existing serving behavior.

## Verification

The offline replay completed 504 two-second classifier forwards.
Every pooled frame result reproduced its earlier raw score to within
0.0002 logit. Native PCM hashes matched all 101 delivered recordings.
The scalar-only verifier checks the original values and evidence
clocks against the three prior receipts, confirms the 12 operating
points and reproduces all reported decisions. The local Seagate archive
retains the source replay, point and score commitments. [Scalar
receipt](evaluation-panels/audio-context-frame-transfer-2026-11-13.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_context_frame_transfer \
  docs/evaluation-panels/audio-context-frame-transfer-2026-11-13.json
```
