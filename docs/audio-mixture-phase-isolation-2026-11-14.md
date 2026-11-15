# Does speaker isolation preserve evidence of a short voice edit?

A separate voice detector previously pointed to four edited recordings at
exactly the same times as their matched genuine mixtures and isolated
backgrounds. This study compares four inputs to that detector to see
whether the short edited words can be recovered from the competing
voices. [Earlier detector results](audio-second-classifier-overlap-2026-11-14.md)

## Comparison

The fixed −5 dB panels supply 48 AMI and 48 ICSI deliveries. Each contains
16 short synthetic word replacements, 16 matched genuine resampling
edits, eight untouched mixtures and eight isolated backgrounds. The four
viewing methods use the same source clock.

- **Original mix.** Reuse the one-second responses from the prior test.
- **Exact target.** Subtract the exact integer background stem from the
  mix. This is a diagnostic available only because the benchmark has
  isolated stems. An isolated background has no target in this view.
- **Blind separation.** Split overlapping four-second windows. Match
  the two estimated voices across the two-second overlap by correlation
  and crossfade the matching estimates. Score both anonymous streams
  when their levels are measurable. Discard a separation when its
  ordering or polarity cannot be established on the overlap.
- **Mixture phase.** Use the same estimated streams to assign a share
  of each time-frequency bin to each stream. Apply those shares to the
  original mix's complex spectrum to preserve its phase wherever a bin
  is retained. The two reconstructed sources add back to the original
  mix at its native sample positions.

The separator publisher marks its original weights as Apache-2.0. The
pinned ONNX conversion separately carries Apache-2.0 and states the
original source. The publisher used a mix of open and private training
data. Overlap with these benchmark recordings is unknown. Local scoring
uses pinned voice detection weights and speech activity weights. All
audio, including the estimated streams, stays local.
[Original weights](https://huggingface.co/alibabasglab/MossFormer2_SS_16K)
· [Pinned ONNX conversion](https://huggingface.co/mhavo/mossformer2-ss-16k-onnx/tree/3be7f95237b5121caf41dec4c62b0af341a640af)
· [Voice detector asset review](commercial-model-audit.md)

All views use one-second windows every half second, with at least half
of the window occupied by detected speech. The strongest classifier
response across the eligible streams and windows can produce one location
per view. AMI genuine reader-group maxima and isolated backgrounds
set each view's separate cutoff before ICSI scoring. A useful location
covers at least half of the annotated edit and has at least 0.30 temporal
intersection over union. The comparison also checks whether that exact
range was flagged on matched ordinary audio or the isolated background.
AMI and ICSI provide separate backgrounds with the same foreground
readers, words and generators.

For diagnosis, compare each estimated stream against the known target
over the exact edited samples using scale-invariant signal-to-distortion
ratio (SI-SDR). Record whether the stream with the strongest classifier
response is also the closest to the target. The known target and edit
bounds enter this retrospective check only. Both practical methods
score anonymous streams.

## Local results

Each panel has 16 labeled edits, 24 ordinary mixtures and eight isolated
backgrounds. Useful locations also appear in the next column when the
view did not select the same range in any of that edit's matched controls.

| Panel | View | Useful locations | Useful without the same control location | Ordinary flags | Background flags |
| --- | --- | ---: | ---: | ---: | ---: |
| AMI development | Original mix | 0 | 0 | 3 | 0 |
| AMI development | Exact target | 4 | 4 | 3 | Unassessed |
| AMI development | Blind separation | 0 | 0 | 0 | 0 |
| AMI development | Mixture phase | 0 | 0 | 3 | 0 |
| ICSI transfer | Original mix | 4 | 0 | 15 | 3 |
| ICSI transfer | Exact target | 4 | 4 | 3 | Unassessed |
| ICSI transfer | Blind separation | 0 | 0 | 0 | 0 |
| ICSI transfer | Mixture phase | 4 | 1 | 16 | 2 |

The exact target recovered the same four replacements from three readers
in both panels. It missed eleven and selected an unrelated moment on
one more in each panel. Blind separation located zero and assessed 47
AMI and 46 ICSI deliveries. Both separated views lost coverage on the
same files. The two ICSI omissions were isolated backgrounds.

The best anonymous blind stream improved median SI-SDR over the edit
by **4.53 dB on AMI** and **0.43 dB on ICSI**. The mixture-phase stream
improved it by **1.60 dB** and **1.77 dB**, respectively. Both acoustic
measurements choose the stream closest to the known target after the
fact. On ICSI the highest-scoring blind stream was closest to the
target in **8/16** edited files. The mixture-phase ranking had the
same count. The classifier often followed a different estimated voice.

The one ICSI mixture-phase location without an exact control match was
**2.500–3.500 s** on a Chatterbox replacement at **2.610–3.520 s**. That
view flagged 16 genuine mixtures from six reader groups and two isolated
backgrounds. The practical methods remain evaluation-only. The low
recall on exact foreground speech makes a short-edit classifier trained
on matched clean and noisy examples a higher priority than a general
audio cleanup step.
[Numeric receipt](evaluation-panels/audio-mixture-phase-isolation-2026-11-14.json)

The independent recount verified all 96 original input hashes, every
estimated waveform hash, the four operating points, and the flagged
ranges and paired-control counts. In 93 deliveries with two retained
mixture-phase streams, the decoded sum recovered the original waveform
to within (10^{-5}) on the original sample clock. The shared one- and
two-second scoring code also reproduced its previous results exactly
on four separately replayed AMI and ICSI inputs.

## Replay

The numeric receipt includes only model responses, sample coordinates,
source commitments and acoustic quality measurements.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_mixture_phase_isolation review \
  docs/evaluation-panels/audio-mixture-phase-isolation-2026-11-14.json
```

For native replay, prepare the committed AMI and ICSI PCM with the
[second detector's source replay](audio-second-classifier-overlap-2026-11-14.md#replay).
Use the pinned ONNX file with ONNX Runtime 1.24.3 plus the app's numerical
dependencies. Use the same speech activity and voice detector snapshots
as the second detector replay. The reference run used the CPU, and all
model and PCM digests are checked before their data is used.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.audio_mixture_phase_isolation prepare \
  ami "$run/ami-views" "$pcm_ami" "$separator_onnx"
backend/.venv/bin/python -B -m scripts.audio_mixture_phase_isolation measure \
  ami "$run/ami" "$run/ami-views" "$snapshot" "$speech_activity_weights"
backend/.venv/bin/python -B -m scripts.audio_mixture_phase_isolation prepare \
  icsi "$run/icsi-views" "$pcm_icsi" "$separator_onnx"
backend/.venv/bin/python -B -m scripts.audio_mixture_phase_isolation measure \
  icsi "$run/icsi" "$run/icsi-views" "$snapshot" "$speech_activity_weights" \
  --development "$run/ami/scored.json"
backend/.venv/bin/python -B -S -m scripts.audio_mixture_phase_isolation combine \
  "$run/ami/scored.json" "$run/icsi/scored.json" "$run/receipt.json"
```
