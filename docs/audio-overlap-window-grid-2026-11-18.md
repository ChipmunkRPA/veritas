# Shorter shared intervals for noisy Opus speech edits

The earlier 750 ms scan offered a correctly positioned window for eight of
75 labeled edits under simultaneous speech. In three of those eight cases,
the best eligible pair included a companion window with zero edited samples.
The weaker window contained a median 0.57% edited foreground energy across
the eight cases. This motivates separating the placement of the two score
inputs from the time range they jointly support.
[Previous exact-foreground comparison](audio-exact-foreground-2026-11-18.md)

## Fixed comparison

Place 750 ms windows at exact 250 ms steps on the native 16 kHz decoded
sample timeline. A window needs at least half its samples covered by the
existing speech mask. Compare the original sparse starts and denser starts
with two supported windows 500 ms apart. Compare emitting both full windows
with emitting only their shared 250 ms range. A fourth arm pairs windows
250 ms apart with a 500 ms shared range. For overlapping shared ranges,
keep the one with stronger minimum support, breaking ties by earlier start.
The reported shared region describes where the two inputs overlap.

All four arms use the same two inherited decision points. The paired windows
overlap, so their responses are correlated. Their minimum gives a way to
rank shared ranges. A location must cover at least half its labeled interval
and reach a temporal intersection over union of at least 0.30. Each predicted
range can match one label. A location clears its paired controls when the
matched genuine upload and the room recording have no competing range at that
label. This test uses the distinction between authenticity scores and temporal
support examined by recent speech localization studies. It uses the project's
existing pinned assets and locally retained recordings.
[Zero-shot localization study](https://arxiv.org/abs/2609.10051)
· [Controlled temporal anchor study](https://arxiv.org/abs/2610.04479)
· [Source and asset terms](audio-isolation-by-codec-2026-11-18.md)

The first panel retains 11 matched source groups with 75 edits from six
voice-generator families. The transfer uses four different foreground groups
and three different room sessions. It includes 30 visible edit labels, of
which 26 end inside the upload. Its three generator families recur from the
first panel. Both use exact existing Opus deliveries. The transfer carrier
clock and decoded bytes were reproduced from their originals before scoring.
All roles use the same schedule within a source group. Clip labels enter only
the location check. The comparison retains three role types per group
including edited mixtures, genuine mixtures and room-only recordings.

At either fixed point, the first panel requires at least three additional
locations across three different groups that clear the controls. The transfer
requires at least two across two groups. The panels allow two and one
additional genuine-file alerts respectively. Both require baseline clear
locations to be retained and zero additional room-only alerts.

## Results

Counts show **located / clear of controls** for edits and the number of files
flagged in each control group. The first panel contains 11 genuine and 11
room-only files. The transfer contains four of each.

| Decision point and window rule | First panel edits / 75 | First panel genuine · room | Transfer edits / 30 | Transfer genuine · room |
| --- | ---: | ---: | ---: | ---: |
| Stricter, sparse full windows | 2 / 2 | 2 · 0 | 0 / 0 | 0 · 0 |
| Stricter, dense full windows | 8 / 4 | 8 · 0 | 3 / 3 | 1 · 0 |
| Stricter, dense 250 ms shared range | 1 / 0 | 8 · 0 | 1 / 1 | 1 · 0 |
| Stricter, dense 500 ms shared range | 3 / 2 | 7 · 3 | 3 / 3 | 2 · 0 |
| Wider, sparse full windows | 3 / 2 | 2 · 0 | 0 / 0 | 0 · 0 |
| Wider, dense full windows | 10 / 5 | 9 · 0 | 4 / 2 | 1 · 0 |
| Wider, dense 250 ms shared range | 1 / 0 | 9 · 0 | 1 / 1 | 1 · 0 |
| Wider, dense 500 ms shared range | 3 / 2 | 7 · 4 | 4 / 4 | 2 · 0 |

Across the first panel, the dense full windows could geometrically reach 59
of 75 labels, compared with eight for the sparse schedule. At the wider
point they added four clear locations across three groups and lost one
earlier clear location. They also added seven genuine-file alerts. The
500 ms shared intervals at the same point found four clear locations in
three groups on the transfer and two of four genuine files also flagged.
All four clear transfer locations were inside the uploaded audio. Room-only
alerts appeared for the closer-paired arm on the first panel. The 250 ms
shared range reached only 22 labels there and found none clear of controls.

No arm passed the fixed decision requirements on either panel. A shorter
shared range still depends on the same overlapping source responses. The
test measured the benefit of more candidate placements and its accompanying
false alarms. The results support keeping this schedule in evaluation.

The first run admitted 1,360 of 1,401 candidate windows. The second admitted
478 of 492. Shared regions and file alerts can be replayed with only the
public receipt and earlier timeline ledger. The replay also verifies speech
admission, operating points, and the exact earlier decisions at common
window starts. The receipt includes anonymous sample positions, speech
duration counts, scalar scores and study commitments. Original recordings
and derived voice representations stay local.
[Anonymous comparison receipt](evaluation-panels/audio-overlap-window-grid-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_overlap_window_grid \
  docs/evaluation-panels/audio-overlap-window-grid-2026-11-18.json.gz
```
