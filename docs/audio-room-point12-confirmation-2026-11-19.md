# Twelve-second confirmation of the room-calibrated speech point

The lower point recovered three additional clean edits and two additional
competing-speech edits. It also added nine ordinary clean alerts, sixteen
ordinary competing-speech alerts and four isolated-background alerts.
Both frozen advancement gates failed. Keep the current app behavior.

This tests the point-only arm from the
[preceding room-transfer comparison](audio-room-layer-short-transfer-2026-11-19.md).
Its earlier two useful delivery additions shared one physical edit and
cost one ordinary alert. The larger ordinary cost in this confirmation
changes the deployment judgment.

## Frozen inputs and comparison

Select one retained MLS host per each of eight reader/donor groups by a
fixed hash order. Move its insertion to a newly qualified 960 ms clock,
at least 750 ms clear of the previous edit. Clock selection uses only
amplitude and existing background annotations. Reuse two exact licensed
stock synthesis fragments per host. The resulting sixteen family/host
events have new physical placements and delivery bytes. Their recordings,
fragments and background sources appeared in earlier project studies.

The eight host readers and eight dedicated genuine donor readers have
zero declared-ID overlap with either the eight BBC point readers or the
24 readers in the preceding room comparison. Independent replay uses the
exact privacy-hash mappings from the original source selectors. Cross-corpus
identity aliases and model-training overlap remain unresolved.

Each event repeats in clean audio, dominant real traffic and dominant
recorded overlapping speech. The noise target is −10 dB, measured on the
untouched target. Four retained captures supply two traffic and two AMI
meeting backgrounds. Every competing-speech target has at least half
annotated overlap. These are held background captures relative to the
BBC point context. The experiment adds no new measured foreground room
response and supplies no population claim about rooms.

Each host has five matched ordinary states in every condition. They are
untouched audio, same-reader and different-reader genuine splices, and
two local resampling round trips. Every substitution has the same fade
and local RMS-matching policy. Sixteen isolated-background deliveries
contain the exact scaled beds used with each host. The full panel has
184 deliveries, with 48 synthetic, 120 ordinary and sixteen backgrounds.
Repeated conditions and control states are correlated observations.

Deliver all inputs as twelve-second, 16 kHz mono PCM16 WAV. The retained
MLS foregrounds originated in Opus. The delivered lossless carrier
preserves their qualified decoded bytes. It restores no information lost
in that earlier encoding. The replacement words are unaligned with the
host wording and carry no voice-cloning claim.

Run the actual app's confirmed extended two-second scan and original
adjacent 750 ms review as the incumbent. Keep every selected and displayed
confirming interval. The candidate adds the same original short-window
responses above **4.382393928394829**, fixed before this confirmation from
the earlier genuine-only BBC role. The classifier, speech gate, original
point, window grid and confidence remain unchanged. No second point or
condition-specific admission is fitted here.

## Closed results

Each condition has sixteen synthetic events and forty ordinary controls.
A useful interval covers half a known edit and has at least 0.30 temporal
intersection over union.

| Condition | Useful, incumbent → candidate | Missed, incumbent → candidate | Ordinary alerts, incumbent → candidate | Background alerts, incumbent → candidate |
| --- | ---: | ---: | ---: | ---: |
| Clean | 12 → 15 /16 | 4 → 1 /16 | 0 → 9 /40 | — |
| Traffic | 0 → 0 /16 | 16 → 16 /16 | 0 → 0 /40 | 0 → 0 /8 |
| Competing speech | 2 → 4 /16 | 14 → 12 /16 | 5 → 21 /40 | 0 → 4 /8 |

The three clean additions span three reader groups and one synthesis
family. The two noisy additions span both families in one reader group.
Together they cover five distinct physical events in four reader groups.
Their five matched ordinary states are clear at the added useful clock.
Ordinary alarms at other clocks remain consequential.

The candidate adds twelve off-target synthetic intervals in clean audio
and 28 with competing speech. Their added union durations are 7.75 and
17.25 seconds across those sixteen files per condition. Newly highlighted
ordinary time totals 11.25 seconds clean and 29 seconds with competing
speech. All previous incorrect intervals remain present.

The nine new clean ordinary alerts span three reader groups and include
two untouched files. Five lie wholly outside the matched target. The
sixteen new competing-speech ordinary alerts span four groups and include
three untouched files. Twelve lie wholly outside the target. The four
new isolated-background alerts are scaled copies of one meeting capture,
so they represent one underlying background source.

Traffic has six unassessed synthetic deliveries, sixteen unassessed
ordinary deliveries and eight unassessed background-only deliveries.
The zero traffic flags therefore provide limited specificity evidence.
Clean and competing-speech inputs have no unassessed deliveries under
either arm. An unassessed interval supports no original-file conclusion.

The clean gate required four added events, gains across both synthesis
families and a smaller ordinary and misplaced-time cost. The broad gate
also required gains in both noisy conditions, broader reader support and
no newly flagged isolated backgrounds. Both remain failed. This confirms
a larger sensitivity/false-alarm tradeoff than the preceding six-second
panel. The point stays in research and adds no customer evidence or
overall confidence weight.

## Independent verification and privacy

A separate native implementation reconstructed every delivery, checked
35,328,000 delivered sample values and 8,478,720 exact foreground sample
identities outside edits, and replayed all 184 WAVs through FFmpeg. The
worst target SNR error was below 0.00013 dB. The scorer streamed one WAV
at a time, keeping the new scratch package to a few megabytes.

The closed independent recount checked 4,232 brief windows, 736 six-second
source excerpts and 7,912 scheduled window/admission bindings. Its own
incumbent selector, adjacent-window selector, paired counts and both
gates match exactly. Eight authored controls passed. The scorer made
1,510 authenticity calls on 1,509 distinct inputs, including an exact
repeat, in about twelve minutes on two CPU threads.

A local-name collision initially prevented serialization of the independent
review receipt after its recount assertions had passed. The repair renames
that list. The original reviewer and its frozen hash are retained. No
scoring, point, decision rule or counted result changed.

The public companion aggregate contains scalar summaries and whole-study
commitments. It excludes reader identities, individual offsets and hashes,
private paths, audio and response vectors. Its standalone reviewer validates
the pinned arithmetic and rejects eight authored alterations.

```sh
python3 -B -S backend/scripts/review_audio_room_point12_confirmation.py \
  docs/evaluation-panels/audio-room-point12-confirmation-2026-11-19.json \
  --self-test
```

Exact native and model replay requires the retained source custody. Source
preparation and permitted assets are documented in the
[held-reader study](audio-short-window-reader-holdout-2026-11-18.md),
[MLS release](https://www.openslr.org/94/),
[AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) and
[voice-model publisher](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector).
This trial uses retained source media and weights. Generation, product
behavior and outside contact remain unchanged.
