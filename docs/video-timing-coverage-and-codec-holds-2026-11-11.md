# Short video timing coverage and re-encoded picture holds

## Timing coverage in the app

The voice and mouth timing check now shows the source-clock interval when it
measures one speech window. If fewer than four windows support the comparison,
the finding says that a change in timing was unassessed. The measured window
remains available as local context. It supplies no confidence for AI origin
or authenticity.

The local replay exercised the production timing adapter on all 176 retained
three-second GRID views from 16 recordings across eight source components.
The same 176 views were used in an earlier serving pilot. This replay used one
largest detected face per native picture and did not rerun the serving tracker,
fusion, or payment path. It checked each whole file, all 13,200 native RGB
pictures and their integer presentation timestamps, and all 8,386,048 signed
16-bit audio samples with exact origins.

| Retained condition | Views | Views with one measured window | Views eligible for a timing change |
| --- | ---: | ---: | ---: |
| Original recordings | 16 | 14 | 0 |
| Speech-conditioned mouth redraws | 32 | 32 | 0 |
| Expression-driven mouth redraws | 32 | 29 | 0 |
| Ordinary picture or soundtrack edits | 96 | 91 | 0 |
| All views | 176 | 166 | 0 |

Every soundtrack admitted at most one full offset-search window. The zero
timing flags are a coverage result. These clips cannot measure sensitivity to
a timing change. A native changed-pixel mask establishes that a mouth was
redrawn. The separate movement audit describes how often each redraw achieved
the requested timing. [Construction](grid-native-clips-v149-result.md) ·
[Achieved movement](grid-mouth-motion-v158-result.md) ·
[Earlier full serving pilot](grid-serving-completion-result.md)

GRID is distributed under CC BY 4.0. The retained generator branches use the
publishers' permitted weights and the previously audited replacements for
restricted face components. Publisher demonstration media were excluded.
LipSync-A is offered under CC BY-NC-SA 4.0 and did not supply this commercial
evaluation. [GRID publisher record](https://zenodo.org/records/3625687) ·
[Speech-conditioned renderer terms](https://github.com/TMElyralab/MuseTalk/blob/0a89dec45a0192b824e3cf4daf96c239440c5ed8/README.md#disclaimerlicense) ·
[Expression renderer terms](https://github.com/KlingAIResearch/LivePortrait/blob/9b294b3d0536135442ea73cb01e6cb3ca7029dd3/LICENSE) ·
[Adapted renderer audit](mouth-sequence-v4-evaluation.md) ·
[LipSync-A publisher card](https://huggingface.co/datasets/AnsonShe/LipSync-A)

## Approximate picture holds after delivery encoding

A separate frozen pilot compared the shipped exact-RGB check with an anchored
picture-similarity rule. It paired three CREMA-D recordings with authored
0.60-second and 1.52-second holds. Each pair used the same delivery recipe.
The matched test files used a 25 Hz picture clock and a six-frame maximum
keyframe interval. Lossy H.264 and VP9 used YUV 4:2:0. The encodes omitted audio
because this comparison measured picture holds.
Twelve VCD2 webcam recordings, each with unchanged, ordinary chroma and
ordinary recompression delivery, supplied independent recording controls.
A localization was counted when its half-open range overlapped the known
hold at intersection over union of at least one half. A flag on any ordinary
view counted as review burden.

| Condition | Views | Exact localized or flagged | Approximate localized or flagged |
| --- | ---: | ---: | ---: |
| Lossless authored holds | 6 | 6 | 4 |
| H.264 CRF 23 authored holds | 6 | 0 | 4 |
| H.264 CRF 37 authored holds | 6 | 0 | 4 |
| VP9 CRF 42 authored holds | 6 | 0 | 4 |
| CREMA-D ordinary recordings across delivery recipes | 12 | 0 | 3 |
| VCD2 ordinary webcam deliveries | 36 | 0 | 12 |

The 15 ordinary flags span five of 15 recording groups. The candidate
frequently found a naturally quiet stretch and sometimes picked it over the
authored hold. It stays outside the product. The exact check remains an
inspection cue for repeated pictures within its opening-frame budget.
[CREMA-D publisher terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) ·
[VCD2 publisher terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA) ·
[Webcam delivery verification](video-vcd2-delivery-v206-result.md)

The candidate used the same native prefix budget as the exact check, with
at least 0.50 seconds of verified pictures and a maximum PTS step of
0.125 seconds. It area-resized each decoded picture to 64×36, converted it
to YCrCb, and compared each picture with the first using channel weights
0.6/0.2/0.2. It accepted a global mean difference of at most 1.8 and a
maximum tile mean of at most 4.0 over a 4×4 tile grid. It required
a changed flank on each side within 0.36 seconds, using global mean at
least 2.5 or tile mean at least 6.0. The longest qualifying span won.
No cutoff was fitted to these clips. This is a small single-machine
method comparison. [FFmpeg's fixed-reference freeze method](https://www.ffmpeg.org/doxygen/trunk/vf__freezedetect_8c_source.html)

For one recovered H.264 hold, the offline verifier independently matched the
0.840–1.600 s review interval, the complete source clock, and native PNG
pixels from 0.680 s, 1.200 s, and 1.600 s. The receipt verifies the
before, during, and after pictures. The overlap metric evaluates the
localization separately. All 72 planned views completed and passed their
input checks. The optional macOS timing wrapper could not read its final
system counters after the completed receipts were written, so no peak
memory figure is reported. [Portable interval verifier](../backend/scripts/verify_native_video_interval.py) ·
[Aggregate results and receipt commitments](evaluation-panels/video-timing-coverage-and-codec-holds-2026-11-11.json)
