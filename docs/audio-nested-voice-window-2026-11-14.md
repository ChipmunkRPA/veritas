# Narrowing a short voice alert under competing speech

The comparison asks whether a one-second response inside an existing
two-second alert can point more precisely to the authored edit. It leaves
the current alert decision fixed. Across 48 synthetic replacements in two
background corpora, the narrower readout located five edits that the original
reported range could not locate under the same overlap rule. The five
deliveries represent two readers. Eighteen already flagged ordinary AMI
deliveries would also receive narrower ranges. The rule remains experimental.

## Source preparation

Select eight utterances from the PhonemeDF publisher prefixes before scoring.
Each genuine utterance has a complete matching word sequence and sampled word
boundaries in both the MeloTTS and Chatterbox TTS archives. The one- and
two-word replacements span 0.33–0.91 seconds. The matched ordinary controls
resample the original words through the same duration changes. The target
and each matched control share one normalization gain and background within
each group. Outside the changed range, the quantized PCM matches the unedited
delivery sample-for-sample. [PhonemeDF release and terms](https://zenodo.org/records/19522979)
· [MeloTTS model card](https://huggingface.co/myshell-ai/MeloTTS-English)
· [Chatterbox model card](https://huggingface.co/ResembleAI/chatterbox)

Choose eight six-second background excerpts from two AMI meetings by manual
word timing, outside the exact captures used in the earlier variable-boundary
trial. At least half of each edited interval overlaps annotated speech from
two or more background participants. Deliver both a 0 dB and a −5 dB mix
with untouched and genuine-resampling controls. The eight isolated
backgrounds provide their own genuine controls. [AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

For the second background panel, reuse the exact eight foreground groups.
Match them one-to-one to eight different ICSI excerpts at −5 dB using
annotations and the same minimum overlap, with no inference output in the
selection. The three selected ICSI meetings contain no shared declared
participants. All eight foreground readers and both AMI meeting sessions
have prior project exposure. The two panels contribute 136 deliveries and
130 distinct submitted PCM inputs. None of those exact PCM inputs appeared
in the earlier contribution study. The ICSI test measures transfer to a new
recording domain while holding foregrounds fixed.
[Qualified ICSI intake](audio-icsi-overlap-intake-2026-11-14.md)

## Fixed scoring and results

Run the actual five-window PCM speech review with its existing point. For an
alert, rank the dense one-second candidates wholly contained by the existing
two-second window and use the point from the earlier development bank.
Retain the existing range if no contained candidate meets that point. The
same speech gate and exact source-sample geometry apply. A useful location
has at least 0.30 intersection over union and covers at least half the known
edit. All 136 deliveries had an assessed speech window. The experiment
reuses the commercially reviewed detector assets. [Prior dense operating point](audio-input-contribution-transfer-2026-11-14.md)
· [Model asset terms](commercial-model-audit.md)

| Background | Synthetic deliveries | Existing alerts on synthetic | Useful ranges before → after | Ordinary alerts | Ordinary alerts narrowed |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMI, 0 dB | 16 | 6 | 0 → 2 | 11/24 | 9 |
| AMI, −5 dB | 16 | 6 | 0 → 2 | 9/24 | 9 |
| ICSI, −5 dB | 16 | 1 | 0 → 1 | 0/24 | 0 |

Neither route alerted on any of the 16 isolated background deliveries. With
AMI speech, both generators moved from **3.000–5.000 s** to **3.500–4.500 s**
for the same **3.550–3.920 s** edit, at both mix levels. The ICSI gain came
from a different reader. The one-second result **1.000–2.000 s** contained
the **1.140–1.520 s** Chatterbox edit. The original range was
**1.000–3.000 s**. The ICSI panel also had 15 of 16 synthetic
deliveries without a current alert. A nested cue cannot recover those
missed alerts. Independent one-to-one event counting reproduced all alert
and edit-localization totals from the archived ranges.

## Replay

Rebuild the counts locally without weights or source files. The ledgers
retain only source commitments, score values and sample positions.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.audio_nested_voice_window_transfer review \
  docs/evaluation-panels/audio-nested-voice-window-ami-2026-11-14.json
python3 -B -S -m scripts.audio_nested_voice_window_transfer review \
  docs/evaluation-panels/audio-nested-voice-window-icsi-2026-11-14.json
```

For native scoring use the exact publisher cache layout in the
[MeloTTS source census](audio-melo-short-source-census-2026-11-13.md#native-replay)
and the local archives named by the runner. Write to distinct new output
directories. The initial AMI run uses the runner checked in at `193ca47a`.
For the ICSI comparison, add `--icsi "$icsi_scratch"` after acquiring the
original sample ranges as described in the ICSI intake.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_nested_voice_window_transfer measure "$run" \
  --melo "$melo" --aligned "$new_cache" --earlier "$old_cache"
```
