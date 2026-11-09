# Matched audio generation and complete source panel

All **464 foreground recordings and 48 backgrounds** are now qualified for the
next four-second audio confidence experiment. The 64 planned matched synthetic
sources generated successfully and passed normal app decoding. No detector,
VAD, new confidence fitting or serving-default change has run on this panel.
These are input-integrity results, not improved detection accuracy.

| Source | Total recordings | Recordings per role |
| --- | ---: | ---: |
| Genuine speech | 224 | 56 |
| Cloned speech | 80 | 20 |
| Expressive synthesis | 96 | 24 |
| Matched stock synthesis | 64 | 16 |
| Speech, noise and music backgrounds | 48 | 12 |

The four roles remain fitting, confidence calibration, method selection and
final confirmation. Each role has 116 foregrounds and four backgrounds of each
kind. All 512 source nodes retain the original 157 connected ancestry groups;
none crosses roles. This includes genuine/synthetic pairs sharing an intended
script. Source-group separation covers known project metadata and byte/sample
connections; it is not proof of no pretraining overlap or biometric identity.

## Actual generation and decoding

The qualified pinned Apache-2.0 stock generator rendered 64 complete source
prompts with 16 publisher-verified stock profiles. All complete phoneme chunks
and source tokens were preserved. A same-seed repeat of the first source was
sample-exact, making **65 forward calls and 64 study sources**. Model state and
profile inputs stayed unchanged. Offline checks recorded no network attempts.

The native mono float32 outputs are 5.05–19.35 seconds long, totaling 820.075
seconds. Every source exceeds the four-second minimum. There were no replaced,
padded, repeated-to-length or otherwise ineligible outputs. No native samples
exceeded the signed-16-bit amplitude range.

The first v79 generation attempt stopped at an unavailable optional audio-file
dependency, before model import or generation. Its failure log and bounded
profile remain intact. The prospective v80 successor changed only the storage
container to installed NumPy with exact sample-byte readback. It retained the
same runtime, complete phoneme plan, source order, seeds, profiles, model and
CPU math, and was published before generation. It did not install an unreviewed
dependency or alter detection settings.

The v81 qualification wraps all 64 native arrays in float WAV containers and
requires exact native sample readback. Each WAV then passes the existing app
decoder, recording its explicit 16 kHz signed-16-bit conversion and complete
source timeline. Native and production sample hashes remain distinct. All
448 previously qualified original files were rehashed. Actual text, byte and
PCM connections reproduce the corrected source-role partition.

All four generation stages passed in 160.35 seconds total. Maximum sampled
child-tree RSS was 4,497,113,088 bytes; minimum observed available RAM was
4,207,542,272 bytes. The complete qualification passed in 9.79 seconds, with a
maximum sampled 124,928,000 bytes. No guard, sampling error or resource-limit
observation occurred. These are sampled bounds, not guaranteed peak-memory
measurements. All 17 focused source/chunk/native-container/ancestry tests and
lint passed locally. A separate count check verified 224 genuine and 240 AI
foregrounds, 116 foregrounds per role and the unchanged protected calibration.

## What happens next

Prepare the fixed clean, noise, music and overlapping-speech variants; publish
the exact inference and combined-calibration method before new detector calls.
Retain the current method, the useful complementary pair and other plausible
combinations on the same inputs. Compare both wrong directions, missed AI,
inconclusive/unassessed cases, calibration quality, runtime and memory. A small
extra error is a measured tradeoff, not automatic rejection. Fit and calibrate
on their assigned roles; keep final confirmation separate from selection.

This panel still covers short English speech. It does not qualify sub-four-
second utterances, intelligibility, exact spoken words or general real-world
accuracy. Expressive stock profiles and music were seen in project history;
new stock profiles and genuine/clone foreground identities were absent from
that recorded history. Shared backgrounds and matched scripts create dependence
within roles. Preserve these limits in later uncertainty estimates.

Methods and aggregate receipts: [v80 protocol](audio-matched-generation-v80-protocol.md),
[v79 preserved stop](evaluation-panels/audio-matched-generation-v79-stop.json),
[v80 result](evaluation-panels/audio-matched-generation-v80-result.json),
[v81 protocol](audio-complete-sources-v81-protocol.md),
[v81 source summary](evaluation-panels/audio-complete-sources-v81.json) and
[v81 bounded profile](evaluation-panels/audio-complete-sources-v81-profile.json).
The prior [source-role qualification](audio-confidence-source-qualification-v75.md)
retains permission, history-exposure and ancestry-correction details. Source
media, texts, individual identifiers/pairings, voice features and learned
artifacts remain private. No payment, workflow, outreach or hosting change.
