# Training a detector for short synthetic speech replacements

The speaker-isolation study found some useful locations when the
detector received the exact clean target. Competing speech and practical
separation erased much of that localization benefit. This trial trains
a separate frame-level detector on partial edits of known duration.
[Speaker-isolation results](audio-mixture-phase-isolation-2026-11-14.md)

## Source and fitting roles

The source archive is the pinned CC BY 4.0 PhonemeDF release. Genuine
reads are attributed to LibriSpeech. The publisher supplies corresponding
MeloTTS and Chatterbox renditions with native word positions. Eleven
declared readers contribute eight utterances each, using four edits in
each duration stratum. Individual readers and recording sessions stay
within one role. The source freeze excludes the exact foreground
utterances from the recent nested AMI/ICSI study and exact keys in the
earlier v9 source plan. The available reader IDs had previous project
exposure. Any model pretraining exposure is unknown.
[PhonemeDF release and terms](https://zenodo.org/records/19522979)
· [LibriSpeech](https://www.openslr.org/12/)
· [Prior source commitments](audio-melo-short-source-census-2026-11-13.md)

Six readers enter fitting, two enter calibration and three enter
assessment. The fitting and calibration backgrounds come from different
AMI sessions. Held mixtures use the three pinned ICSI rooms, with the
foreground crop chosen to place the edit under annotated overlap. Within a
matched set, each version uses an identical gain and identical background
samples. At least half of each labeled edit overlaps annotated concurrent
background speech. The overall mix ratio is −5 dB foreground to
background. It does not describe the level of each word. Repeated
background excerpts are counted separately from the number of recordings
and rooms. The 88 prompts use 23 different fitting excerpts, 16
calibration excerpts and 12 held excerpts. In total there are
1,144 planned deliveries and 1,120 distinct PCM files.
[AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
· [ICSI source replay](audio-icsi-overlap-intake-2026-11-14.md)

Each prompt has an unchanged version, an ordinary splice from the same
reader, two ordinary round-trip resamplings of the original words, two
synthetic versions of those same words, and an isolated background.
The same gain matching and 20 ms fades apply to the edited arms. Every
dry foreground has a matched noisy version. Their PCM is identical
after exact integer subtraction of the common background. The source
generator labels come from the publisher.

## Detector

A 3,626-parameter temporal convolutional network reads 64 log-Mel bands in
25 ms frames every 10 ms. It learns two frame responses. One learns
whether a frame lies inside the edited interval, and one learns
the vicinity of the two known boundaries. Its objective also ranks
the edited region above genuine edits at the same clock and penalizes
changes in those margins between the dry and noisy views. This is
a local implementation with no code or checkpoint borrowed from
the cited research. [Boundary and segment localization](https://arxiv.org/abs/2609.25822)
· [Partially spoofed speech and frame-level supervision](https://arxiv.org/abs/2204.05177)

Fit two separate networks. The first sees only MeloTTS replacements
and is evaluated on Chatterbox, and the second reverses those roles.
The fit never uses the held generator's resampling control either.
Each network uses independently computed fitting normalization.
After fitting, genuine calibration controls set two cutoffs before
held inference. The strict cutoff is the highest genuine score. The
wider cutoff is the 95th percentile of ordinary foreground scores,
bounded below by the highest background-only score. Both use a strict
greater-than comparison.

The scorer reads one upload at a time and ranks speech-admitted
windows of 0.50, 0.75 and 1.00 seconds. Each proposed location
combines interior and endpoint responses. Evaluation counts a range
as useful when it covers at least half of the labeled edit and
achieves 0.30 temporal intersection over union. Genuine splices,
resampling, untouched clips and isolated backgrounds serve as matched
checks for misleading locations. The incumbent runs its normal
two-second PCM review on held noisy mixtures.

## Held results

All 24 synthetic edits, 72 genuine mixtures and 24 isolated-background
deliveries in each noisy generator comparison were assessed. The two
comparisons reuse the same three held readers and the same underlying
prompt and room bank. An isolated excerpt can recur at different
foreground-dependent gains.

| Held generator | Check | Useful edit locations /24 | Locations also absent from matched genuine and background results | Genuine flags /72 | Isolated background flags /24 |
| --- | --- | ---: | ---: | ---: | ---: |
| Chatterbox | Current PCM review | 2 | 0 | 32 | 0 |
| Chatterbox | Trained on MeloTTS, either cutoff | 4 | 1 | 48 | 12 |
| MeloTTS | Current PCM review | 2 | 0 | 32 | 0 |
| MeloTTS | Trained on Chatterbox, strict | 0 | 0 | 0 | 0 |
| MeloTTS | Trained on Chatterbox, wider | 0 | 0 | 15 | 2 |

The qualifying Chatterbox location was **2.900–3.400 s**, overlapping
the edit at **3.110–3.590 s**. Adding that detector to the current
review would add four useful locations, including three whose exact
ranges also occurred in matched controls. The combination would add
29 genuine flags and 12 isolated-background flags. The wider cutoff
on held MeloTTS added six genuine flags and two background flags,
with no added useful location.

Neither network usefully located a dry edit in either cutoff
comparison. The Chatterbox-held model flagged 30/72 dry genuine
controls at its strict cutoff. The MeloTTS-held model flagged
none at its strict cutoff and 3/72 at the wider cutoff. A
separate inspection of the frozen scores compared each synthetic clip
against all three matched genuine clips at their strongest reported
locations. The synthetic clip had the highest score in just **2/24**
noisy Chatterbox groups and **5/24** noisy MeloTTS groups. In dry
audio the counts were **1/24** and **5/24**. Both models remain
evaluation-only. The poor ordering of matched genuine and
synthetic clips points to the learned signal and its transfer,
even before choosing a cutoff.
[Numeric receipt](evaluation-panels/audio-short-edit-detector-training-2026-11-14.json)

## Independent replay

The published numeric receipt preserves source and model commitments,
operating-point scores, selected sample ranges and separate results
per held generator. Recount it with standard-library Python.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_short_edit_detector_training review \
  docs/evaluation-panels/audio-short-edit-detector-training-2026-11-14.json
```

For native replay, prepare the retained PhonemeDF byte ranges, AMI
captures and pinned ICSI excerpts described in the
[earlier replay](audio-nested-voice-window-2026-11-14.md#replay).
Keep generated waveforms and trained weight files in a local run
directory. The public receipt contains neither waveforms nor per-frame
features. The training step reproduces both weight files from their
fixed seed and stores their hashes alongside the operating points.
The reference Apple Silicon CPU run used PyTorch 2.13.0, NumPy 2.2.6,
Transformers 5.15.1 and safetensors 0.8.0. A second run reproduced
the exact source freeze, both trained files and candidate measurements.
A separate recount checked all 1,144 PCM commitments, verified
sample identity outside the edits, and reproduced the operating
points and held location counts. The native selection expects the
historical v9 source plan in the local `--history` directory.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.audio_short_edit_training_sources "$sources" \
  --melo "$melo_ranges" --aligned "$phonedf_ranges" --earlier "$legacy_ranges" --icsi "$icsi_excerpt_bank"
backend/.venv/bin/python -B -m scripts.audio_short_edit_detector_training study \
  "$sources" "$run/model" "$voice_weights/silero_vad.jit"
backend/.venv/bin/python -B -m scripts.audio_short_edit_detector_training incumbent \
  "$sources" "$run/incumbent.json" "$voice_weights"
backend/.venv/bin/python -B -S -m scripts.audio_short_edit_detector_training combine \
  "$sources" "$run/model/candidate.json" "$run/incumbent.json" "$run/receipt.json"
```
