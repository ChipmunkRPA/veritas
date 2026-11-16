# Genuine speech profiles for locating short edits under loud conversation

The test asks whether ordinary speech alone can teach a detector which
parts of a six-second recording deserve review. The November
one-class study fits a regularized covariance on frozen speech
representations and shows that a useful source-domain ranking can
degrade across delivery conditions. This local implementation changes
the unit to a short interval and learns separate clean and loud-mix
profiles. It uses the project's previously pinned encoder without
importing the study's implementation or weights.
[November genuine-only study](https://arxiv.org/html/2611.11465v1)
· [Pinned encoder and weight terms](audio-wavlm-v13-protocol.md)

## Fit and test design

1. Fit on 48 genuine prompts from the fitting readers. The dry and
   loud recordings each include the untouched and genuine-splice
   version, for 192 deliveries. For each speech-supported window,
   use the mean and standard deviation of 20 fixed projections at
   layers 3, 9, and 12. Fit separate standardized covariances by
   window length and acoustic condition with 15% identity shrinkage.
2. Examine 400, 600, 800, and 1,000 ms windows with starts every
   100 ms. Require detected speech in at least half of each window.
   Score each layer by the smaller robustly scaled distance from
   the dry and mixed profiles. Compare the early layer, the average
   of the three layers, and the early response above both equal-width
   speech intervals with 100 ms guards. Each rule selects one range.
3. Set the 90th and 75th percentile points on the maximum
   per file for 128 separate ordinary deliveries. Each cutoff uses
   the highest reading from 16 isolated calibration excerpts as
   its lower bound. A follow-up fits an additional reference on 48
   isolated excerpts from the fitting capture. It uses the closest
   of all three profiles and repeats the same separate calibration.
4. Compare the fixed rules with the local two-second check. Also
   try each as a fallback after a current non-flag. A useful range
   must cover at least half the actual edit and reach 0.30 temporal
   intersection over union. Recount matched ordinary and room
   alerts at the same edit clock.

The 346 held deliveries contain 337 distinct PCM inputs. The PhonemeDF
panel uses 24 project-held prompts with staged Chatterbox and MeloTTS
replacements and matched genuine splices and resampling. Three
ordinary deliveries share exact PCM with another role. The earlier SINE
panel contains 32 pairs. The **additional 21 readers** come from
unused eligible pairs in the same eight pinned source prefixes. The
selection excludes the previous SINE rosters and checks declared reader
commitments against the PhonemeDF panel. Both SINE panels use the same
six independent AMI room excerpts at 5 dB above the original voice
level. The stereo transfer reads the previously measured mono mix of
60 files, with Piper and Kitten inserts in Aachen and Huddersfield room
responses. These are controlled mixtures. Encoder pretraining exposure
and cross-corpus reader aliases remain unknown.
[PhonemeDF sources and terms](audio-short-edit-detector-training-2026-11-14.md#source-and-fitting-roles)
· [SINE source terms](https://huggingface.co/datasets/PeacefulData/SINE)
· [AMI source terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
· [Stereo source and generator terms](audio-stereo-channel-replication-2026-11-15.md#source-and-room-design)

## Held results

The highest isolated calibration reading set both points for all
six candidate arms. Each assessed all 137 edited deliveries and
173 ordinary foreground deliveries. All methods assessed 34 of
the 36 isolated-room deliveries. Six exact room inputs recur across
the SINE panels. The two-second check had 49 edited files with a
flag. Forty-one of those locations missed the known edit.

| Profiles and readout | Useful locations /137 | Ordinary flags /173 | Isolated flags /36 |
| --- | ---: | ---: | ---: |
| Current two-second check | 8 | 59 | 0 |
| Clean and mixed, early layer | 0 | 0 | 0 |
| Clean and mixed, layer average | 0 | 0 | 2 |
| Clean and mixed, nearby contrast | 0 | 4 | 4 |
| Include isolated conversation, early layer | 0 | 1 | 2 |
| Include isolated conversation, layer average | 0 | 0 | 3 |
| Include isolated conversation, nearby contrast | 0 | 0 | 2 |

Four of the eight current useful ranges had a clear matched control
at the same clock. On the 21 new SINE pairs the current check found
zero useful ranges, reported seven misplaced ranges, and flagged six
ordinary files. All six candidates found zero useful ranges there.
The narrower geometry could fit all 21 edits with the known clock,
compared with four on the two-second grid, before the speech gate.
The strongest pre-cutoff nearby-contrast proposal under the clean
and mixed profiles was useful on three of the 21. Better geometry
alone left most edits below other sounds in the same file.

A diagnostic recomputed the two points without the isolated-audio
floor. With the 90th percentile and the two-profile nearby contrast
as a fallback, useful locations on the new SINE panel rose from
0/21 to 1/21. Ordinary flags rose from 6/21 to 8/21, and one of the
six isolated rooms flagged. The lower early-layer point also located
three additional edits in Huddersfield, all with a matched-control
flag at the edit clock. Adding the isolated-conversation profile
left the frozen useful counts at zero. Both profile variants stay
in evaluation. [Two-profile results](evaluation-panels/audio-local-speech-normality-2026-11-16.json.gz)
· [Isolated-profile results](evaluation-panels/audio-local-speech-normality-isolated-2026-11-16.json.gz)

## Independent replay

The receipts record source and weight commitments, fitted-profile
hashes, exact timestamps, and scalar scores. The portable reviewer
verifies the original comparison receipts and source labels, then
recalculates cutoffs, matched alerts and the fallback outcomes.
Its counts agree on Python 3.12 and 3.14. The raw recordings,
projected speech features and fitted profiles stay local.

```sh
export PYTHONPATH=backend
module=scripts.audio_local_speech_normality
panels=docs/evaluation-panels
python3 -B -S -m "$module" review "$panels/audio-local-speech-normality-2026-11-16.json.gz"
python3 -B -S -m "$module" review "$panels/audio-local-speech-normality-isolated-2026-11-16.json.gz"
python3 -B -S -m "$module" review "$panels/audio-local-speech-normality-2026-11-16.json.gz" --foreground-only-points
```

Rebuild the [fitting PCM](audio-short-edit-detector-training-2026-11-14.md#independent-replay),
[retained SINE inputs and eight prefixes](audio-short-edit-union-holdout-2026-11-15.md#reproduce),
and [stereo source inputs](audio-stereo-channel-replication-2026-11-15.md#replay).
The additional SINE intake reproduced all 48 PCM deliveries and its
source plan byte for byte on a second pass. The two-profile native
inference reproduced its complete receipt byte for byte. A separate
sample-set overlap calculation also reproduced the frozen flag and
location counts in both receipts.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
encoder=models/research/wavlm-v13
weights=models/weights/audio-upgrade
"$runner" -B -m "$module" fit "$phoneme" "$encoder" "$weights/silero_vad.jit" "$run/fit"
"$runner" -B -m "$module" fit "$phoneme" "$encoder" "$weights/silero_vad.jit" "$run/isolated" --extend "$run/fit"
"$runner" -B -m "$module" fresh "$prefixes" "$retained_sine" "$run/fresh"
"$runner" -B -m "$module" score "$phoneme" "$retained_sine" "$run/fresh" "$run/fit" "$encoder" \
  "$weights" "$run/two-profiles.json.gz" --stereo "$stereo"
"$runner" -B -m "$module" score "$phoneme" "$retained_sine" "$run/fresh" "$run/isolated" "$encoder" \
  "$weights" "$run/three-profiles.json.gz" --stereo "$stereo" --prior-readout "$run/two-profiles.json.gz"
```
