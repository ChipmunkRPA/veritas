# Testing when overlapping speech should make a voice flag inconclusive

The current voice detector flagged 12 of 16 known genuine recordings when mixed with an overlapping room cut in an earlier test. This study asks whether an independent acoustic observation can identify some unreliable AI flags without discarding correct synthetic detections. The rule can only change an existing AI direction to inconclusive. [Earlier room result](audio-independent-background-transfer-2026-11-11.md)

## Fixed measurement

On the unchanged three-second mono input, compare the harmonic patterns in 64 ms windows on a 20 ms grid. Fit the strongest pitch candidate to prominent frequency peaks. Mask its harmonic neighborhood, then fit a second pitch from the remaining peaks. Keep the pair only if the pitches and peak support are sufficiently distinct and the two patterns continue for at least three adjacent frames. The cue measures concurrent harmonic structure. Two nearby pitches, unvoiced speech, reverberation and music can affect the result. The cue does not identify speakers or establish the origin of the sound. [Harmonic cancellation method](https://www.ee.columbia.edu/~dpwe/papers/Klap03-multif0.pdf) · [Spectral peaks and non-peak regions](https://interactiveaudiolab.github.io/assets/papers/taslp2010-duan-pardo-zhang.pdf) · [Local implementation](../backend/scripts/audio_harmonic_overlap.py)

Select an acoustic point on 16 native development windows, allowing no more than one marker in each room's four minimal-overlap controls. Among those options prefer more marked high-overlap examples, then fewer minimal-overlap marks, then the stricter duration. Use archived incumbent and independent Spectra scores from the 77-view earlier experiment to select separate score points for the disagreement rule and its conjunction with the acoustic cue. Neither may lose a clean synthetic flag or more than one synthetic flag during point selection. Measure acoustic-only, independent disagreement, and their conjunction separately. An option without a viable development point cannot withhold a flag. [Pinned independent model and rights](audio-independent-background-transfer-2026-11-11.md)

## Source-separated comparison

The confirmation contains 14 LibriSpeech utterances from seven speakers and nine synthetic utterances from three distinct groups in each of three generator families. Their declared voices, groups and foreground fingerprints are disjoint from the earlier point-selection bank. Each foreground appears clean and in a −5 dB mix from each of two AMI rooms. Each room contributes four different interfering-speech cuts. The eight background-only and sixteen untouched native windows bring confirmation to 93 views. The natural cuts are annotation-selected on a fixed grid, and their participants are disjoint from the two development meetings and the other known prior meeting groups. Four native windows per room have at least 1.2 seconds of simultaneous annotated words, and four have at most 0.1 seconds. Each cut has at least 2.45 seconds of annotated words from one or more participants. Word timings can omit breaths and nonverbal sounds. The source banks had earlier preparation use. The two confirmation room groups, their sample intervals and the new mixtures are separate from this method's development, and checkpoint training overlap is unknown. [Foreground source review](audio-short-edit-source-plan-v169.md) · [AMI downloads](https://groups.inf.ed.ac.uk/ami/download/) · [AMI CC BY 4.0 terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The decision criterion requires at least three and at least 30% of the incumbent genuine false flags withheld, from at least two source groups. Preserve at least 90% of synthetic flags, lose at most one overall, and preserve all clean synthetic flags. Require at least six assessed genuine mixtures in each room. For the acoustic routes, also require markers on at least four of eight native high-overlap windows, with at least one per room and at most two markers across the eight minimal-overlap windows. Public results report aggregate counts. The sample-clock evidence and audio stay local.

## Local results

At the shortest development point, the acoustic cue marked two of eight high-overlap windows and six of eight minimal-overlap controls. All qualifying points had zero high-overlap marks. The tie-break selected 640 ms. Neither score-based abstention rule found a point that removed a known false flag while satisfying the synthetic protection requirements.

| Held condition | Genuine false AI / assessed | Correct synthetic AI / assessed | Synthetic inconclusive | Synthetic leaning original |
| --- | ---: | ---: | ---: | ---: |
| Matched clean | 0/14 | 9/9 | 0 | 0 |
| First recorded room at −5 dB | 1/14 | 5/9 | 3 | 1 |
| Second recorded room at −5 dB | 2/14 | 2/9 | 7 | 0 |
| Untouched native microphone cuts | 0/10 | — | — | — |
| Isolated interfering speech | 1/8 | — | — | — |

The acoustic rule marked zero of eight held high-overlap windows and zero of eight minimal-overlap windows. Before applying the duration point, the acoustic spans totaled 1.94 seconds across native windows. They shared 1.39 seconds with the 13.44 seconds of annotated overlap. Six of 16 untouched microphone windows were unassessed by the current voice detector. All constructed mixtures and isolated backgrounds passed speech admission.

None of the three rules withheld any of the four genuine false flags or any of the 16 correct synthetic flags. The high-overlap evidence and decision criteria failed. This test demonstrates the limits of the harmonic cue on natural rooms, including where two controlled overlapping tones produced a correctly localized interval. [Exact aggregate and independent audit](evaluation-panels/audio-harmonic-overlap-2026-11-11.json)

The current route made 118 classifier forwards, including stability probes, and the independent CPU detector made 21 forwards, including its exact repeat. A separate standard-library audit reconstructed the operating points, checked all 186 PCM hashes and 40 native cuts against the official microphones, remixed all 46 constructed controls, rebuilt 20 graph input hashes, and reproduced the decision arithmetic.

## Reproduce

Use the retained Spectra recovery directory with its original `ami/` and `comparison/` subdirectories and the pinned checkpoint assets from that study. Supply the two official confirmation WAV files in a separate directory. The [runner](../backend/scripts/audio_harmonic_overlap_transfer.py) derives the room and source selections from the pinned annotations and original qualification receipts. The [reviewer](../backend/scripts/review_audio_harmonic_overlap.py) needs the resulting local directory, the confirmation WAV directory and the earlier development WAV directory.

```sh
export PYTHONPATH=backend
python -m scripts.audio_harmonic_overlap_transfer freeze "$run" --recovery "$recovery" --ami "$ami" --assets "$assets"
python -m scripts.audio_harmonic_overlap_transfer fit "$run"
python -m scripts.audio_harmonic_overlap_transfer baseline "$run"
python -m scripts.audio_harmonic_overlap_transfer confirm "$run" --assets "$assets"
python -m scripts.audio_harmonic_overlap_transfer readout "$run"
python backend/scripts/review_audio_harmonic_overlap.py "$run" "$ami" "$recovery/ami"
```
