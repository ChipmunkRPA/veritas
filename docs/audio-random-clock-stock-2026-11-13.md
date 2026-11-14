# Speech localization with randomized edit times and competing voices

The first stock-voice test placed both possible edits near the centers
of the historical sparse windows. This follow-up places edits across
the entire valid insertion clock and brings in new host and donor
readers, a different Piper checkpoint, a different Kitten stock voice,
and three new background captures. The analysis compares placement,
false flags, and whether a background-reduced copy recovers otherwise
missed edits. [First stock-voice comparison](audio-stock-voice-transfer-2026-11-13.md)

## Frozen inputs

The test uses 24 new original recordings from eight host and four
ordinary donor readers, with 12 separate books, acquired through ten
bounded English MLS test-shard ranges. The prior stock roster shares
no selected reader, book, audio commitment, or full transcript. The
earlier v9 and v172 rosters also have no matching declared reader IDs,
audio commitments, or full transcripts for these selections. The
Piper Norman model was trained from scratch on public-domain speech.
Kitten Luna uses the previous Apache-2.0 checkpoint and a separate
stock voice. Both generators use new authored prompts and no host voice
clone. This measures stock-voice replacements and shares generator
architecture lineage with the previous test. [MLS corpus and CC BY 4.0
release](https://www.openslr.org/94/) · [Piper Norman model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/norman/medium/MODEL_CARD)
· [Piper voice author and rights](https://brycebeattie.com/files/tts/)
· [Kitten checkpoint and grant](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8)

Each host supplies untouched, same-reader splice, different-reader
splice, and two generated-splice variants. Every splice uses the same
960 ms clock, 20 ms fades, and gain procedure within its source group.
Assign host groups to eight equal-width start strata and use a salted
hash to draw a start sample within each stratum before inference. The
resulting starts span 0.622–4.980 s and the last edit ends at 5.940 s.
The results use the exact sample coordinates, with seconds rounded for
readability. A useful selection spans at least half the edit and meets
0.30 temporal IoU.

Three additional participant-disjoint AMI meeting groups provide the
six-second, native 16 kHz competing sound. The actual edit intervals
have 260–960 ms of annotated overlapping background words. Conditions
repeat at 0 and −5 dB. Waveform subtraction verifies the same PCM
masker on every matched version. The difference is confined to the
planted span in all 64 authored edits. There are 32 generated deliveries,
48 matched ordinary deliveries, and three isolated backgrounds. Only
16 distinct generated edits and eight host groups underlie the repeats.
[AMI corpus and CC BY 4.0 release](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Frozen checks

The existing optional short review examines five two-second windows starting at
0, 1, 2, 3, and 4 s. The historical sparse review uses only **1–3 s**
and **3–5 s**. It is reported both at its original point and at the
live point to isolate how the smaller grid changes the outcome. All
routes retain the published voiced-window admission rule.
At the frozen coverage criterion, the full grid admits a geometrically
useful window for all eight positions. The sparse grid can reach six.
The 0.622–1.582 s and 4.980–5.940 s edits are outside its useful overlap
at either decision point.

A separate severe-noise route keeps any live flag. For −5 dB deliveries
and isolated backgrounds, apply the pinned MIT GTCRN streaming
checkpoint over the submitted six-second PCM and measure signal-energy
reduction. Where the original did not flag and energy decreased by at
least 6 dB, run the live five-window review and its unchanged point on
the reduced copy. The detector reads only the delivered mixture or its
processed copy. The true foreground is used separately to measure
signal-to-distortion improvement. The enhancer's clock matches the
input, and its recurrent cache resets at each recording. No result was
used to tune either rule. [Published enhancement checkpoint and MIT grant](https://github.com/Xiaobin-Rong/gtcrn/tree/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73)
· [Earlier traffic-sound transfer](audio-traffic-enhancement-transfer-2026-11-11.md)

## Held results

Each cell reports valid locations out of 16 AI-splice deliveries,
wrong-window flags on those deliveries, and flags out of 24 ordinary
controls. All 83 deliveries yielded admitted windows. All routes gave
zero flags on the three isolated backgrounds.

| Route | 0 dB | −5 dB |
| --- | ---: | ---: |
| Existing five-window review | 4/16 · 1 wrong · 3/24 ordinary | 1/16 · 1 wrong · 3/24 ordinary |
| Earlier sparse point | 6/16 · 1 wrong · 7/24 ordinary | 2/16 · 2 wrong · 6/24 ordinary |
| Sparse at the existing point | 1/16 · 0 wrong · 2/24 ordinary | 0/16 · 2 wrong · 3/24 ordinary |
| Five windows with background reduction | 4/16 · 1 wrong · 3/24 ordinary | 4/16 · 4 wrong · 11/24 ordinary |

At the same point, the five-window review recovered four additional
useful placements over the sparse grid across both noise levels. The
earlier sparse point added three useful placements overall and seven
ordinary flags compared with the five-window review. The result
supports retaining full-clip coverage. The smaller grid can still be
useful when an exact interval is supplied by independent evidence.

The severe-noise enhancement recovered both generator versions of the
**0.622–1.582 s** edit in **0–2 s**. It also placed a Kitten insert at
**3.941–4.901 s** in **4–6 s**. All three additional useful indications
came from presentations using the same recorded AMI meeting. The same
route added eight ordinary flags, bringing the total to 11 across five
host groups, and added three wrong-window flags. Median signal energy
decreased by 7.01 dB. Median foreground SI-SDR improved by only 0.90 dB
against the exact clean source. This suggests attenuation alone did not
reliably isolate the intended speaker. The optional short-review
windows and confidence calculation retain their existing behavior.

## Verification

The frozen run completed 551 voice-classifier passes and 16,544
streaming enhancement frames including the exact repeat check. A
separate native check verified the 83 committed waveforms, 64 matched
splice clocks and exact masker reuse. The largest error from the
intended host-to-background ratio was 0.00001411 dB. The local Seagate
archive retains the frozen waveforms and associated scripts. The
public receipt contains source commitments, scores, and sample
positions without audio or text. [Scalar receipt](evaluation-panels/audio-random-clock-stock-2026-11-13.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_random_clock_stock \
  docs/evaluation-panels/audio-random-clock-stock-2026-11-13.json
```
