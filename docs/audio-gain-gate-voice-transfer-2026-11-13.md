# Speech decisions after a gain-assisted retry

The retry added **zero correct synthetic detections** across 261 deliveries.
It scored three previously skipped quiet synthetic windows, and all three
returned inconclusive. The live voice check keeps its existing admission
and confidence calibration.

## Comparison

Both studies tried the gain view only when the original speech gate did
not admit the three-second window. The second view increased the level by
up to 6 dB without clipping. When it reached the existing half-window
speech requirement, the current voice detector and stability check
scored the **original** samples. Already-scored files kept their saved
directions. This experiment follows the earlier observation that gain
can increase speech coverage in some quiet natural recordings.
[Earlier speech-coverage study](audio-speech-admission-transfer-2026-11-13.md)

| Comparison | Deliveries | Skipped before → after | Added correct synthetic flags | Added genuine or background AI flags |
| --- | ---: | ---: | ---: | ---: |
| Environmental sound and competing voices | 193 | 20 → 20 | 0 | 0 |
| Quiet speech and matched room mixtures | 68 | 18 → 15 | 0 | 0 |

The 193-file replay reopened 116 environmental deliveries and 77
competing-voice deliveries and reproduced every prior primary decision.
The quieter panel used 32 distinct voices, with eight genuine voices and
24 published synthetic voices across three generator families. Each voice
had a quiet clean version at −36 dBFS RMS and a matched −5 dB mixture
with an AMI room excerpt. Four isolated room excerpts served as controls.
The room clips had zero annotated words. That label does not establish
silence or exclude every audible voice. The exact quiet mixtures were
new. Both their component banks had earlier project exposure.
[Retained noisy inputs](audio-paired-loud-noise-2026-11-13.md) ·
[Voice-source rights and provenance](audio-short-edit-source-plan-v169.md) ·
[Room source and license](audio-speech-admission-transfer-2026-11-13.md)

On the quiet panel the retry admitted one clean synthetic excerpt and
two noisy synthetic excerpts from the expressive family. All remained
inconclusive. Of the 24 synthetic clips in the quiet mixtures, two leaned
AI, one leaned original in error, 13 were inconclusive after the retry,
and eight remained unassessed. The eight genuine quiet mixtures had two
original-leaning and six inconclusive results. Four isolated backgrounds
remained unassessed. These counts describe the matched test conditions.
They cannot establish accuracy on new speakers, noise, or generators.

## Application and replay

The ordinary audio result now links from an inconclusive or unassessed
headline to the recorded speech coverage. Chrome opened the link on a
completed noise-only analysis and showed two skipped windows, exact
source timestamps, and zero voice scores. Customer confidence continues
to use only the admitted original-input checks.

The two public replays recount anonymous decisions and source
commitments. Neither ledger includes audio, voice embeddings, or model
files. Native inference uses the pinned commercially permitted local
weights and the exact source banks from the linked preparation reports.
[Archived replay ledger](evaluation-panels/audio-gain-gate-voice-transfer-2026-11-13.json) ·
[Quiet replay ledger](evaluation-panels/audio-quiet-gain-transfer-2026-11-13.json)

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -B -S -m scripts.audio_gain_gate_voice_transfer review \
  docs/evaluation-panels/audio-gain-gate-voice-transfer-2026-11-13.json
python3 -B -S -m scripts.audio_quiet_gain_transfer review \
  docs/evaluation-panels/audio-quiet-gain-transfer-2026-11-13.json
backend/.venv/bin/python -B -m scripts.audio_gain_gate_voice_transfer freeze "$archive_trial" --archive "$competing_bank"
backend/.venv/bin/python -B -m scripts.audio_gain_gate_voice_transfer measure "$archive_trial" --archive "$competing_bank" --environmental "$environmental_bank"
backend/.venv/bin/python -B -m scripts.audio_quiet_gain_transfer freeze "$quiet_trial" --manifest "$held_voice_manifest" --cache "$ami_replay"
backend/.venv/bin/python -B -m scripts.audio_quiet_gain_transfer measure "$quiet_trial"
```
