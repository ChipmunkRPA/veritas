# Original-audio boundaries across four edit lengths

The variable-duration cue nominated useful ranges on five of eight held expressive speech edits before applying an operating point. At the frozen point it flagged none. At two matched false groups it located one edit, as did the current check. Their useful locations came from different edits, and the variable cue also flagged an isolated-room control. This trial supplies no basis for adding the cue to customer audio analysis. [Aggregate results](evaluation-panels/audio-variable-boundary-transfer-2026-11-12.json)

## Sources and method

Eight held English edits use exact previously unused generated archive recordings from eight declared ElevenLabs v4 profiles and six source subsets. The verified package includes the publisher's original audio bytes and generation sidecar under its pinned archive hash. Each edited clip has matched ordinary same-speaker and other-speaker controls, with 16 distinct genuine LibriSpeech readers across the host and other-speaker roles. The genuine clips use previously unanalyzed post-prefix spans from the retained source catalogue. The generated archive, voice profiles, and genuine readers have prior project exposure. Exact raw bytes, normalized scripts, and decoded PCM fingerprints were screened against the bound history. The selected generator inputs come from one model family. [Generated recording terms and provenance](https://huggingface.co/datasets/laion/elevenlabs-v4-five-language-voice-acting) · [LibriSpeech publisher](https://www.openslr.org/12/) · [Source preparation](audio-calibration-data-preparation-v68.md)

The four edit lengths are 0.40, 0.64, 0.80, and 1.04 seconds, each present once in each of two rooms. The rooms and their eight participants are separate from the earlier paired-boundary trial. Both rooms had prior unrelated project exposure. This study freezes ten fresh native capture intervals, including two isolated-room controls, with no overlaps among the selected intervals or the bound earlier native cuts. For each edited group, the matched original mono files share exact background samples, outside-edit samples, RMS matching, and 20 ms fades. Recorded background speech is 5 dB louder than the unedited host across the full six-second clip. Annotated words from at least two competing voices overlap each known edit for 260–800 ms and cover at least half the edit. [AMI terms and annotations](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The incumbent applies its usual speech gate to the native two-second windows starting at 1 and 3 seconds and uses the existing checkpoint's raw log-odds. Its original fixed point and the earlier 0.80-second paired-boundary point stay frozen. The variable-duration cue evaluates all four lengths using the same adjacent spectral-distribution comparisons at both ends. Each length is standardized using its median and interquartile range on the same 12 exact previously measured genuine controls. The highest standardized eligible interval wins. The variable point is the maximum genuine file response across those controls. This sets all points before held scoring. All arms read the original PCM. No source separation or enrollment enters scoring. [Earlier boundary trial](audio-original-pcm-boundary-transfer-2026-11-12.md) · [Existing checkpoint terms](commercial-model-audit.md)

A useful interval has at least 0.30 temporal intersection over union and covers at least 50% of the edit. The two-second output cannot meet the intersection requirement for a 0.40-second edit even with complete coverage. The 0.80-second cue therefore supplies the primary width comparison for the new shorter candidate. The fixed comparisons use separate operating points. The equal-false-group comparisons describe retrospective ranking at zero, one and two groups. An ordinary splice highlighted in either genuine view counts as a false flag for that source group. Reports also track correctly located ordinary-splice intervals separately to distinguish an observed change from an AI-specific cue.

## Held results

| Comparison | Current native check | Fixed 0.80 s boundary | Variable boundary |
| --- | ---: | ---: | ---: |
| AI flags at fixed point | 2 | 0 | 0 |
| Useful AI locations at fixed point | 1 | 0 | 0 |
| False genuine groups at fixed point | 2 | 0 | 0 |
| Useful AI locations at 0 / 1 / 2 equal false groups | 0 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 1 |
| Useful AI ranges before score points | 2 | 3 | 5 |

The test has eight generated edits, 16 matched ordinary splices, and two isolated-room controls. The current check had one unassessed input. At its fixed point, it usefully located one ordinary other-speaker splice, which counts as a genuine false flag. At two false groups, the variable cue found a 0.80-second generated edit at 0.32–1.12 seconds in one room, reporting 0.56–1.60 seconds. The current check found a different 0.64-second edit at 4.08–4.72 seconds in the other room, reporting 3.00–5.00 seconds. Both boundary arms flagged an isolated room at that retrospective false-group budget. The new cue improved unthresholded placement relative to the fixed boundary. The held outcomes still show difficulty separating the edited locations from room speech and ordinary splices.

## Replay

The current check made 50 CPU forwards on the 52 native windows. The native audit reconstructs every held source waveform, exact component, and decision. The standard-library reviewer independently checks the stored original and component PCM, the native sample clocks, actual annotated background overlap, and the fixed and matched-alarm counts. Neither audit repeats neural inference.

```sh
base=backend/data/independent-evaluation
run="$base/audio-variable-boundary-2026-11-12"
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_variable_boundary_transfer audit "$run"
python3 -B -S backend/scripts/review_audio_variable_boundary_transfer.py "$run" \
  "$base/spectra-recovery-2026-11-11/ami/ami_public_manual_1.6.2.zip"
```

To recreate the run with the retained archives and point receipts, use a new local directory and run `freeze`, `score`, and `report` before both audits. The public aggregate contains no source audio or per-recording results. [Runner](../backend/scripts/audio_variable_boundary_transfer.py) · [Independent reviewer](../backend/scripts/review_audio_variable_boundary_transfer.py)
