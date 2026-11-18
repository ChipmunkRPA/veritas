# Does recurrent-spectrum attenuation help find speech edits?

This local test examined mono Opus files with competing conversation about
10 dB louder than the foreground over the full recording. We looked for new
edit locations and tracked alerts on genuine recordings and isolated room
speech. The current voice check remains in use.

## Method

For each short-time spectrum, find six similar spectra at least one second
away and estimate the recurrent energy from their median. Apply a soft mask
to the submitted complex spectrum, retaining at least 45% of each bin's
amplitude. The altered waveform retains the original sample clock. Detached
recordings and study labels enter only the evaluation. The general idea of
using spectral repetition to estimate background comes from background and
foreground separation research. Component-level deepfake research motivated
the independent test of which speech the mask actually attenuated.
[Recurrent background modeling](https://interactiveaudiolab.github.io/assets/papers/Rafii-Liutkus-Pardo%20-%20REPET%20for%20Background-Foreground%20Separation%20in%20Audio%20-%20Springer%202014.pdf)
· [Component-level audio detector analysis](https://arxiv.org/abs/2610.17617)

The same local voice detector reads unaltered and masked 750 ms windows at
matching positions. Each proposed interval needs support from a second
window beginning 500 ms away. A useful interval covers at least half of the
known edit and has at least 0.30 temporal intersection over union. Each
proposal can match one known edit. Both internal score cutoffs were frozen
from seven genuine source groups and one isolated background. The stricter
cutoff uses the largest genuine control reading and the wider cutoff uses
the second largest. Both are bounded below by the isolated-background
reading. [Existing voice asset terms](commercial-model-audit.md)

## Results

The 26-delivery pilot included eight calibration recordings and 18 held
recordings. At the wider point the prior Opus comparator with its earlier
short-window addition located 3 of 55 edits. Adding the recurrent-mask
readings brought that to 7 of 55, with 4 of 8 genuine files flagged in both
cases and isolated-background flags rising from 0 to 1 of 2. The one added
location with both matched controls raised the same alert on both controls.
The other three added locations lacked matching originals in this panel.
[Prior Opus evaluation](audio-reflected-short-opus-2026-11-18.md)

The second run applied the same mask and cutoffs to 11 same-utterance
triples, each containing an edited recording, its ordinary recording, and
their isolated background. These 33 deliveries use speakers and five AMI
meeting groups distinct from the pilot and calibration. They contain 75
known edits from six generator families, CosyVoice, ElevenLabs, GPT-SoVITS,
LJ-Jets, XTTS v2 and YourTTS. The source archive had earlier project use.
All 33 Opus clocks were verified against the delivered sample counts.
[Matched source preparation](audio-six-ten-context-2026-11-16.md#same-utterance-controls)
· [AMI recording and annotation terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

| Confirmation check | Useful locations /75 | Genuine alerts /11 | Isolated background alerts /11 |
| --- | ---: | ---: | ---: |
| Strict raw | 2 | 4 | 0 |
| Strict masked | 1 | 3 | 0 |
| Strict union | 2 | 4 | 0 |
| Wider raw | 3 | 4 | 0 |
| Wider masked | 4 | 4 | 0 |
| Wider union | 4 | 5 | 0 |

The wider masked check added one controlled location. Its **10.000–10.750 s**
proposal covers the **10.206–10.436 s** LJ-Jets replacement. Both matched
controls stayed clear. At the wider point, masking also newly flagged one
genuine file and cleared one previously flagged genuine file. The union
keeps both genuine flags. The remainder of the pilot's gain did not recur.

The separate component audit applied the masks derived from the Opus mixtures
to each known lossless component. Across 75 edited intervals, median
foreground attenuation was 1.72 dB and median interference attenuation was
1.51 dB. The median change in the foreground-to-interference ratio was
**−0.25 dB** across edits and **−0.57 dB** over the whole edited files. The
recurrence method did not improve isolation of the foreground under competing
speech on this panel. The mask remains an evaluation candidate.

## Verify the comparison

The compact receipt stores source and clock commitments, calibration maxima,
sample positions, admission and pass bits, and component power changes. The
standard-library reader checks the cutoffs, source-role separation, matching,
alerts, and component medians. Source recordings and window-level model
features stay in local evaluation storage. The native-run files are bound by
SHA-256; the command below verifies the published numeric results.
[Compact evaluation receipt](evaluation-panels/audio-recurrent-spectrum-opus-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_recurrent_spectrum \
  docs/evaluation-panels/audio-recurrent-spectrum-opus-2026-11-18.json.gz
```
