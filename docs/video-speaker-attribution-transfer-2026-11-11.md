# Speaker attribution on continuous camera speech

The timing adapter first needs evidence that a visible mouth belongs with the
soundtrack. On the prior cleanest 16-second AMI cut, the camera and headset
correspond to the same participant and the published annotations contain
15.504 seconds of target speech. The automatic gate had 23 possible timing
windows. Its crop-motion correlation peaked at **0.107**, below its 0.20
gate. An independent source check measured actual mouth opening against the
acoustic envelope across the same cut. Mouth opening peaked at 0.363, the
eye-aperture control at −0.039, and frame-to-frame change in mouth opening at
0.165. These are relationships on one known recording. [Prior timing transfer](video-longer-timing-transfer-2026-11-11.md) ·
[AMI camera and microphone mappings](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)

## Candidate comparison

Falcon is an audio-visual speaker separator published in October 2026. It
extracts speech from audio using a sequence of mouth crops. We tested whether
the separated output was more consistent with the source audio when shown
the matching mouth than when shown an upper-face crop or a mouth sequence
from another moment. The publisher checkpoint is 47,156,343 bytes with
11,724,385 parameters, including the frozen visual encoder. The CPU worker
verified a strict weights-only load and bitwise parity of the cached
embedding path on one controlled input. The publisher licenses both the
inference code and the weights under Apache 2.0. Training and demonstration
media were outside this evaluation. [Paper](https://arxiv.org/abs/2610.32749) ·
[Publisher source](https://github.com/JusperLee/Falcon/tree/5c5d4ff6ce99a63742101a3a7992592068df496c) ·
[Publisher model card](https://huggingface.co/JusperLee/Falcon)

A separate anatomical comparison measured the distance between the inner
lips, normalized by face width. It compared the correlation with the audio
envelope to the eye-aperture and wrong-time mouth controls. These comparisons
test a different representation from the current gate's frame-to-frame
motion in a small grayscale mouth crop. [Speech and face-motion study](https://doi.org/10.1371/journal.pcbi.1010273) ·
[Face geometry model review](grid-portrait-geometry-v144-method.md)

The frozen plan used six 16-second continuous cuts from two AMI sessions.
Four came from the earlier timing comparison. Two came from fresh time
intervals in those same sessions, chosen using publisher speech annotations
before extraction and scoring. Every cut supplied five matched views.
The recorded track, a 320 ms local delay over 5–10 s and a constant 320 ms
delay kept their own speaker. An off-screen control used only the headset
from the other session. A mixture combined both exact recordings at equal
whole-cut RMS. [AMI release and license](https://groups.inf.ed.ac.uk/ami/download/) ·
[Research consent](https://groups.inf.ed.ac.uk/ami/corpus/ethicsandconsent.shtml)

The worker verified all 2,400 source pictures and 1,536,000 PCM samples
before inference. It qualified 1,357 native meshes. Each candidate input
needed 50 observed crops at 25 Hz in the tested interval, all shifted and
control sequences, and speech activity in at least half of the corresponding
audio ticks. It filled no missing crops. The working clocks used the actual
camera presentation timestamps and corresponding exact headset samples.
Each window tried visual shifts of −320 ms, zero, and +320 ms. The frozen
separator attribution point required two windows with coherent waveform
gain above −12 dB and a 3 dB advantage over both control crops. The timing
point required the expected shift in all four windows with a 1 dB lead and
passing attribution. These are experimental comparison points.

The face task's first CPU launch stopped at macOS OpenGL initialization
inside the sandbox. The same frozen CPU task then completed with its
required graphics context. One scoring attempt hit a mixed-key JSON
serialization error. The rerun changed the output serialization and
counted only the study forwards. The media, weights, computation, windows,
and comparison points remained fixed.

## Results and decision

| Matched condition | Views | Current speaking-face coverage | Falcon attribution | Anatomical attribution |
| --- | ---: | ---: | ---: | ---: |
| Recorded soundtrack | 6 | 2 | 0 | 0 |
| Local soundtrack delay | 6 | 2 | 0 | 1 |
| Constant soundtrack delay | 6 | 2 | 0 | 1 |
| Unrelated off-screen voice | 6 | 0 | 0 | 0 |
| Exact two-voice mixture | 6 | 1 | 1 | 0 |

The current route counts views with at least one measured timing window.
Each candidate counts views with at least two attributed windows. Each
candidate passed one individual window on an off-screen voice control, so
the zero attributed views there depend on the multiple-window requirement.

The anatomical method picked the expected lag in all four windows of one
locally delayed cut that the current route could not attribute. Only the
two inner windows passed its mouth attribution rule. The second tested
local-delay cut missed the expected lag in all four windows. Across eight
eligible local-delay windows, Falcon and the anatomical method each
picked the expected lag in four. Falcon met the complete timing pattern
in zero views. Its unshifted mixture output improved the known target's
relative energy in five of six measured windows, with a median gain of
10.44 dB. The separation result is specific to these source-bound mixtures.

A newly sampled continuous cut gave the current route 21 measured windows
on the original, 22 on the local delay, and 24 on the constant delay. An
independent lossless remux reran those results on exact picture timestamps
and PCM. The original and local-delay views produced the same
**10.740–12.000 s** timing flag. The local intervention ended at 10.000 s,
so the flag's overlap with the known intervention was zero. The constant
delay gave no timing flag. The offline verifier matched native PNGs at
10.720, 11.360, and 12.000 s in both flagged views.
[Native interval verifier](../backend/scripts/verify_native_video_interval.py)

The 30-view comparison ran 45 visual and 170 audio model forwards on the
CPU in 93 seconds, excluding warmup and the parity check. Neither candidate
established better coverage for genuine speech or reliable localization of
the known timing edit. Both remain offline experiments. The six cuts cover
two recording sessions, and only two cuts supplied the full geometry for
the delayed candidate comparisons. [Aggregate counts and receipt commitments](evaluation-panels/video-speaker-attribution-transfer-2026-11-11.json)
