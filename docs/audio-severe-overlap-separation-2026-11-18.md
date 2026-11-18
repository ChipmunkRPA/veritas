# Two-track separation under competing speech

The earlier known-stem test showed that the recurrence mask could attenuate
the wrong voice when the intended voice was quiet. We tested the released
Kezek separator on delivered mono waveforms, measured whether its tracks
preserved the intended voice, and checked whether interference dominated the
result. Kezek estimates speaker activity over the whole recording and then
uses those estimates to divide the original audio into two tracks. Its
authors report that weak replies and partial speaker swaps remain difficult.
[Earlier known-stem study](audio-stem-mask-ceiling-2026-11-18.md) ·
[Kezek paper](https://arxiv.org/html/2611.16933v1)

## Assets and exact comparison

We ran the MIT-licensed 1.0.1 implementation and Apache-2.0 weights pinned to
revision `a31397b5f637e9e6512bdff1f17dfb9efd2975e2`. The official release and
all three weight files matched published SHA-256 hashes. The code ran offline
on the Mac with matching CPU Torch and TorchAudio 2.8.0. The runner used a
restricted tensor loader for the checkpoints. The authors identify their
training sources in the release card. Open Yap's published agreement permits
commercial model training and deployment subject to its terms. The test used
previously reviewed local benchmark recordings and accessed no Open Yap
recordings or individual speaker data from that dataset.
[Official implementation](https://github.com/Manifestro/kezek) ·
[Pinned weight card](https://huggingface.co/manifestro/kezek-v1/tree/a31397b5f637e9e6512bdff1f17dfb9efd2975e2) ·
[Package release](https://pypi.org/project/kezek/1.0.1/) ·
[Open Yap terms](https://theagenticdatacompany.com/open-yap-1k/license)

- The recurring Opus bank has 33 native-clock deliveries across 11 groups.
  Its 75 edited intervals include 57 at or below −6 dB foreground relative
  to the reference background. At least 10 groups contain simultaneous
  voices within the room microphone track itself. The remaining capture
  also lacks a verified single-talker background. This measures recovery of
  a target against aggregate interference.
- The separate Opus transfer has four matched waveform-verified groups, 12
  deliveries and 30 visible edited intervals. Three labels meet an upload
  edge. The 22 intervals at or below −6 dB include 19 with both original
  edit endpoints inside the upload. Room microphone speaker count is unknown.
  The groups and AMI sessions are separate from the recurring bank. Their
  edited recordings have prior project exposure.
- The two-voice bank has six pairs of separately verified source tracks.
  Each group supplies a six-second edited mix, a matched genuine mix, and
  the competing talker alone. A separate condition reuses these exact six
  groups and lowers the target by a fixed 12 dB before mixing. All six
  attenuated edit intervals are at or below −6 dB and five are at or below
  −10 dB. The speaker and edit coordinates are unchanged across conditions.

Publishers provide the original benchmark files and corpora for research
use. AMI documents participant consent for its recordings. The experiment
reuses the already-published benchmark edits. The receipt contains numeric
measurements without speech, speaker embeddings, derived waveforms,
recording names or participant identifiers.
[Two-voice preparation and terms](audio-two-voice-components-2026-11-15.md) ·
[LlamaPartialSpoof](https://zenodo.org/records/14214149) ·
[LibriTTS](https://www.openslr.org/60/) ·
[AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) ·
[AMI ethical annex](https://groups.inf.ed.ac.uk/ami/corpus/ethics_part_of_orig_ta.pdf)

## Acoustic readout

The model receives the submitted waveform alone. For measurement after
inference, the two exact benchmark references select a single track order
for each file using only samples outside the labeled edits. We use that
choice throughout the file. An apparent switch counts when a different
local assignment wins by at least 3 dB in summed scale-invariant
signal-to-distortion ratio (SI-SDR). With aggregate room interference this
is component assignment disagreement. Every calculation uses the original
16 kHz positions with no alignment shift. Target level is the energy
projection onto the reference target after accounting for the interference
reference. Residual energy includes interference and any distortion.

We also formed complementary masks from each estimate and applied them
to the input spectrum, preserving its phase and sample clock. This follows
the existing 8% per-track minimum mixture mask experiment. It lets us check
whether reconstructing from the original waveform preserves more evidence.
[Earlier waveform reconstruction](audio-overlap-reconstruction-2026-11-16.md)

| Evaluated edit intervals | Median SI-SDR change | Median target level change | Better SI-SDR and target loss ≤3 dB | Target stronger than residual |
| --- | ---: | ---: | ---: | ---: |
| Recurring Opus bank, all 75 | +2.62 dB | −4.86 dB | 20/75 | 16/75 |
| Recurring Opus bank, 57 at or below −6 dB | +4.46 dB | −4.97 dB | 12/57 | 9/57 |
| Separate Opus transfer, 22 at or below −6 dB | +0.23 dB | −7.51 dB | 3/22 | 2/22 |
| Verified two-voice bank, native levels | +1.24 dB | −4.84 dB | 2/6 | 3/6 |
| Verified two-voice bank, target attenuated | +11.52 dB | −6.72 dB | 2/6 | 3/6 |
| Earlier separator on the same attenuated bank | +7.43 dB | +4.99 dB | 3/6 | 1/6 |

SI-SDR can rise even when the preserved target becomes much quieter. The
attenuated two-voice condition illustrates the distinction. The signal shape
improved in aggregate but the median target lost about 6.7 dB, and the
target became stronger than its residual in only half the edits. At native
levels, the local order differed strongly from the fixed order in two of
the six edits. That count was zero in the attenuated condition. The 57
severe recurring intervals had seven strong local component disagreements
and the 22 severe transfer intervals had seven. These differences do not
identify individual room speakers.

In the severe recurring intervals, rebuilding from the input spectrum reduced
the useful retention count from 12 to nine. In the attenuated two-voice
bank the count stayed at two and the number of target-dominant outputs fell
from three to two. The separator emitted two outputs even when given only
the single known competing voice in the controlled bank. The quieter output's
median energy was 17.04 dB below the input. Two output tracks alone provide
no evidence of a second speaker.

The earlier SepFormer checkpoint preserved the intended track's absolute
level in more of these edits. Its target was stronger than its residual in
one of six. Its quieter output on the single-talker recordings was only
3.55 dB below the input in median power, so gating on track energy alone
would also be misleading with this separator. The direct outputs need not
sum to the input. Both models were evaluated on the same hashed 18 files
in this condition. [Pinned SepFormer weights](https://huggingface.co/speechbrain/sepformer-whamr16k/tree/21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5)

The 81 Kezek runs preserved the exact sample count. The summed output matched
the input within a relative RMS error of 0.000001. Median CPU measurement was
about 2.7 seconds for the six-second controlled files and about 5 seconds for
the recurring Opus files. These figures exclude startup and file preparation.
A fresh process reproduced identical two-track outputs and activity results
for an edited, an ordinary and a single-talker file from one attenuated group.

## Diagnostic with the existing authenticity check

The same recurring Opus files also ran through a fixed set of 48 pairs of
750 ms detector windows. It covers eight of the 75 labels under the earlier
matching rule of at least half an edit and 0.30 temporal intersection over
union. Every window used the existing speech admission and both frozen
operating points. The comparison tried both separated outputs with no
choice based on the target stems. There were 160 new forwards on one CPU
thread.

| Frozen operating point | Useful overlapping locations, mixture → with separation | Matched ordinary alerts | Background-only alerts |
| --- | ---: | ---: | ---: |
| Stricter | 2 → 4 | 2 → 3 of 11 | 0 → 0 of 11 |
| Wider | 3 → 5 | 2 → 3 of 11 | 0 → 0 of 11 |

At each point both new locations also coincided with an alert at the same
time in the matching ordinary recording. There were zero additional
locations that cleared those controls. The fixed schedule cannot measure
full-file recall, and the operating points came from the original waveform.
Use the original waveform for authenticity assessment.
[Voice diagnostic receipt](evaluation-panels/audio-severe-overlap-voice-diagnostic-2026-11-18.json.gz)

## Recount

The portable reviewer recomputes the tables using the retained numeric
measurements, original sample locations and local source commitments. No
models or audio are needed. [Numeric receipt](evaluation-panels/audio-severe-overlap-separation-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_severe_overlap_separation \
  docs/evaluation-panels/audio-severe-overlap-separation-2026-11-18.json.gz
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_severe_overlap_voice_diagnostic \
  docs/evaluation-panels/audio-severe-overlap-voice-diagnostic-2026-11-18.json.gz
```
