# Locating short speech edits under a louder voice

The earlier six-second speech check found 11 of 73 known edits inside its
search range when competing speech was 10 dB louder. This study tests
whether two differently prepared views make the same detector more useful
on those recordings. [Earlier six-second comparison](audio-six-ten-context-2026-11-16.md#complete-fresh-comparison)

## What the comparison changes

Run a pinned two-voice separator on the center six seconds of the delivered
mono PCM at 16 kHz. Score its two decoded estimates. In the other method,
derive complementary magnitude masks from those estimates and apply the
masks to the original recording's complex short-time spectrum. A 512-sample
Hann window advances 128 samples at a time. Each mask retains at least 8% of
the original bin and both masks sum to one. The inverse transforms reproduce
the input when added, on the same sample clock. Both methods estimate views
of a mixture and leave speaker identity unresolved. [Pinned Apache-2.0 checkpoint](https://huggingface.co/speechbrain/sepformer-whamr16k/tree/21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5)
· [Soft-mask reconstruction research](https://dafx12.york.ac.uk/papers/dafx12_submission_23.pdf)

Use the same local speech encoder, activity check and short-edit head for
both views. For each allowed interval length, take the strongest reading
across the two outputs. Normalization is fitted separately from the
maximum readings on untouched and genuinely reprocessed prompts. The
wider setting uses the 90th percentile of those control maxima and the
stricter setting uses their maximum. Both settings also use the strongest
background-only reading as a lower bound. The existing local search has
priority for the fallback comparison. [Earlier encoder and weight terms](audio-wavlm-v13-protocol.md)

Recent component-level research compared detectors on the mixed waveform
and found that separately reconstructed speech can distort the features
used for spoof detection. Controlled partial-spoof work also reports
that representing a boundary better can change genuine-edit error rates.
This experiment tests those risks on the same delivered mixtures and
retains genuine processing controls. It uses no code or weights from those
papers. [Component-level study](https://arxiv.org/html/2610.14141v1)
· [Controlled partial-spoof study](https://arxiv.org/html/2610.04479v1)

## Source and clock commitments

- The calibration bank uses 24 prompts from three readers and two AMI
  meeting groups. It contains 24 untouched, 48 genuine resampling and
  24 isolated-background deliveries. Calibration uses no generated edits.
- The main test uses the 120 strongly masked deliveries from the
  prior six-second evaluation and 24 clean untouched controls. The 24
  edited deliveries represent six generator families. Their 73 fully
  in-scope edits are compared with 24 untouched, 24 conventionally
  processed and 48 isolated-background recordings at the same ratio.
- The paired diagnostic uses 14 prior same-utterance edit and original
  pairs plus 14 isolated backgrounds. The edit pairs contain 42 in-scope
  changes. The paired bank includes related source utterances and AMI
  meetings from the main test. The calibration readers and meeting IDs
  are distinct from both test banks. These audio recordings appeared in
  earlier project comparisons. The two separating methods are new to
  these exact inputs. [Matched original source preparation](audio-six-ten-context-2026-11-16.md#same-utterance-controls)
  · [AMI data terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The mixing ratio is set during corpus preparation from whole-track RMS.
It can vary within the inspected six seconds. A useful proposal covers
at least half the labeled edit and reaches at least 0.30 temporal
intersection over union. The portable reader matches each proposal to
at most one edit. A separate diagnostic uses the known clean source
after classification to measure waveform similarity at the original
clock and the detector's performance with the exact clean foreground.
Those references take no part in decisions or calibration.

## Held results under the louder voice

| Comparison | Correct ranges /73 | Untouched flags /24 | Ordinary processing flags /24 | Background-only flags /48 |
| --- | ---: | ---: | ---: | ---: |
| Current longer-file scan plus the six-second check | 11 | 5 | 5 | 0 |
| Six-second check on the mixture alone | 1 | 1 | 1 | 0 |
| Decoded estimates alone, wider | 0 | 2 | 2 | 1 |
| Original spectrum masked into two views, wider | 2 | 6 | 6 | 5 |
| Current scan with a fallback to decoded estimates, wider | 11 | 7 | 7 | 1 |
| Current scan with a fallback to masked views, wider | 13 | 10 | 10 | 5 |
| Current scan with a fallback to masked views, stricter | 12 | 5 | 5 | 3 |

The two additional locations at the wider point come from different
generator families and distinct source readers. Five more edited
files receive highlights that miss the location requirement. The two
methods add no flags on 24 clean untouched controls. The original-PCM
check finds a useful candidate among the best windows of six widths
for 14 labeled intervals before applying its cutoff. The decoded and
masked views have 13 and 12, respectively. The masked view's top-ranked
proposal overlaps eight labeled intervals before applying a cutoff.
Lowering the cutoff would also expose proposals at the wrong times.
These candidates remain in evaluation.

## Exact foreground comparison

On the 14 same-utterance pairs, the masked view found two of the 42
known in-scope edits. It also flagged two matched originals and one
isolated background. The stricter setting found one edit and flagged
one isolated background. None of the useful locations was also flagged
at the same clock on its matched original. The six-second check on the
unmodified mixture and the direct decoded views found zero edits.

Across 28 recordings with an exact clean foreground, the median
source-to-distortion improvement was 13.06 dB for the closer decoded
estimate and 7.90 dB for the closer masked estimate. The clean reference
is used only to measure these gains after scoring. The six-second head
found zero of the 42 edits even when scored on the exact clean track at
its frozen raw-audio operating point, and flagged one exact clean
ordinary track. In the clean edited recordings, the highest-ranked
candidate overlapped five of 42 labels before applying the cutoff.
Better source isolation alone leaves a substantial timing and scoring
gap for this head. [Calibration and per-delivery results](evaluation-panels/audio-overlap-reconstruction-2026-11-16.json.gz)

The model-free event counter reproduced the results for six methods
on all 186 test deliveries. A separate local waveform run reproduced
both estimates and both masks' reconstructed waveforms exactly for
one added edit, one ordinary alert and one background-only alert.
The masked streams' sum differed from the original by less than
0.00000025 normalized amplitude in each of those runs.

## Replay

Place the retained source banks under the directory containing
`audio-six-ten-context-comparison` and `audio-matched-originals`.
The [six-second input and matching guide](audio-six-ten-context-2026-11-16.md#replay)
documents the source archives and matching procedures. Each prepared
PCM is rehashed before reading either view. Files with separated speech
stay in the local scratch directory. The public calculation uses exact
sample offsets, scalar readings, grouping and file commitments.

The separation runner was tested with isolated Torch 2.8.0, TorchAudio
2.8.0 and SpeechBrain 1.0.3. Use the exact checkpoint, model card and
architecture file from the pinned publisher revision. The local loader
constructs the reviewed network and reads tensor weights only. The
scoring phase uses the existing backend environment and existing speech
assets. Neither phase needs network access after provisioning.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
PYTHONPATH=backend python3 -B -S -m scripts.audio_overlap_reconstruction prepare "$bank" "$run/plan.json"
PYTHONPATH="$separation_runtime:backend" "$runner" -B -m scripts.audio_overlap_reconstruction_views \
  "$run/plan.json" "$bank" "$weights" "$run/views" calibration
PYTHONPATH=backend "$runner" -B -m scripts.audio_overlap_reconstruction_score score \
  "$run/plan.json" "$bank" "$run/views" "$bank/audio-six-ten-context-comparison/fit" \
  models/research/wavlm-v13 models/weights/audio-upgrade/silero_vad.jit "$run/calibration.json.gz" calibration
```

Repeat the two commands for `held` and `matched`, saving each score
file under its partition name and adding `--calibration "$run/calibration.json.gz"`
to both scoring commands. Compare another detector's ranges using the
same sample clock and input manifest.

```sh
export PYTHONPATH=backend
"$runner" -B -m scripts.audio_overlap_reconstruction_score collect \
  "$run/calibration.json.gz" "$run/held.json.gz" "$run/matched.json.gz" "$run/result.json.gz"
receipt=docs/evaluation-panels/audio-overlap-reconstruction-2026-11-16.json.gz
python3 -B -S -m scripts.audio_overlap_reconstruction verify "$receipt"
python3 -B -S -m scripts.audio_overlap_reconstruction export "$receipt" "$review"
python3 -B -S -m scripts.independent_review count-events "$review/manifest.json" \
  "$review/truth.json" "$review/plus_mixture_phase.json" "$run/recount.json" --match 0.30 0.50
```
