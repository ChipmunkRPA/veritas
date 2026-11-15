# Stereo speech comparison with recorded room interference

This trial tests whether a second channel can corroborate a weak earlier-copy
match when both microphones heard the same competing room speech. The earlier
stereo test put the extra interference on one channel. Both checks use the
existing local waveform comparison and require a clean, aligned earlier copy
of the speech under review. [Earlier separate-channel test](audio-clean-speech-stereo-long-2026-11-15.md)

## Source and comparison

The room audio comes from the AMI Meeting Corpus, released under CC BY 4.0.
Two microphones from the same synchronized array, channels 01 and 05, supply
16 disjoint twelve-second windows from meetings ES2011a and IS1004a. Window
selection uses the published word times before scoring. A qualifying
window has at least 2.4 seconds of annotated words in
each half and recorded speech across at least half of each known synthetic
edit. Candidate windows run in twelve-second steps from 60 to 588 seconds.
A source-specific hash chooses among eligible unused windows.
[AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) ·
[Microphone and capture details](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml) ·
[AMI annotations](https://groups.inf.ed.ac.uk/ami/download/)

Each room window receives dry speech from one of 16 retained foreground
groups. Twelve come from the ICSI source bank and four from the stock-voice
bank. The latter uses MLS speech. Every replacement clock and clean companion
is retained from the earlier study. The 32 synthetic replacements use the
same four generator families. Half the replacements occur after six seconds. The
foreground's amplitude on one side is 55% of its amplitude on the other;
the quieter side alternates within each meeting. Room speech is 5 dB louder
than the whole foreground by joint stereo energy. Each group uses the same
room samples and gain across all versions. The stereo deliveries use
48 kHz AAC-LC. The comparison discards decoded padding and measures ranges
on the earlier 16 kHz source clock.
[ICSI source intake](audio-icsi-overlap-intake-2026-11-14.md) ·
[MLS and stock replacements](audio-random-clock-stock-2026-11-13.md)

Compare the downmix with requiring both channels to flag overlapping ranges,
selecting the channel with the stronger reference match, and the same
selection with at least 0.15 reference correlation in both halves of the
other channel. The last point comes from the earlier one-sided interference
test and stays fixed here. Wrong-reference controls include the exact left
room track, an unrelated foreground, and a reference that shares just one
half. Genuine room-only deliveries supply both a matching room reference
and an absent target reference. These controls distinguish an unsupported
comparison from a comparison that admits the wrong voice or flags an
incorrect range.

## Local results

| Comparison | Correct synthetic edit ranges | Correct genuine splice ranges | Ordinary foreground clips flagged | Flags with the room as the reference | Flags on room audio alone with its matching reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Downmix | 14/32 | 8/20 | 0/40 | 10/16 | 9/16 |
| Both native channels agree | 13/32 | 7/20 | 0/40 | 0/16 | 0/16 |
| Stronger reference match | 14/32 | 8/20 | 3/40 | 2/16 | 0/16 |
| Stronger match plus other-channel support | 14/32 | 8/20 | 3/40 | 2/16 | 0/16 |

Selecting the stronger channel lost none of the useful downmix locations.
It added two misplaced synthetic ranges and one misplaced genuine-splice
range. All three ordinary flags came from the same meeting. Both
stronger-match rules still admitted all 16 exact-background references
for comparison. Every rule declined all 16 unrelated and all 16
half-matching foreground references. All 16 room-only submissions stayed
unassessed when paired with the absent foreground reference.

After scoring, we tested retaining the downmix range only where it
overlapped at least half of the supported-channel range. With equal
720 ms ranges the criterion is symmetric. It kept all 14 correct
synthetic and eight correct genuine-splice ranges, with zero ordinary
flags and one incorrect range on the 16 mixed recordings compared
against the exact background. It returned zero local flags on genuine
room-only submissions. Recounting this additional rule on the previous
one-sided panel kept 32/64 synthetic and 19/40 genuine-splice ranges,
with 4/80 ordinary flags versus 15/80 for the downmix. On the earlier
scaled-and-delayed stereo panel, it lost one correct location from
each edit class and reduced ordinary flags from 12/80 to 3/80. The
room result tests two independent recordings as backgrounds for a
reused foreground roster. The agreement rule stays experimental pending
fresh recordings selected before scoring, especially in cases where
the foreground and room reference both vary across microphones.
[Frozen placements and numeric outcomes](evaluation-panels/audio-stereo-room-transfer-2026-11-15.json.gz)

## Replay

The public receipt stores source hashes, the frozen placements, exact range
ticks and scalar support. It reconstructs all paired outcomes and verifies
that switching references leaves the encoded file unchanged. Python 3.12
and 3.14 produced identical recounts. A separate native input replay
reconstructed all 172 submissions and references from the pinned sources
and verified that all 52 waveform edits changed samples only within their
labeled intervals before encoding. For native
replay, use the same prepared dry foreground banks as the
[earlier AAC study](audio-clean-speech-aac-2026-11-15.md#local-app-and-replay).
Download the publisher's four WAV files and manual annotations v1.6.2
from the AMI download page. Preserve the identifiers in the filenames.
The script checks the exact original bytes before reading any PCM.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
module=scripts.audio_stereo_room_transfer
receipt=docs/evaluation-panels/audio-stereo-room-transfer-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
python3 -B -S -m "$module" review "$receipt" --diagnostic

backend/.venv/bin/python -B -m "$module" freeze "$run/plan.json" \
  --rooms "$rooms" --annotations "$annotations"
backend/.venv/bin/python -B -m "$module" run "$run/plan.json" \
  --rooms "$rooms" --annotations "$annotations" --icsi "$icsi" --ami "$ami" \
  --receipt "$run/room.json.gz"
```
