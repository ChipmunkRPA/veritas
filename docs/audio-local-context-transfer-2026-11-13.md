# Do short noisy voice edits respond differently to surrounding speech?

This local comparison tested a speech representation on the same waveform
with and without wider recording context. The new check localized **0 of
12** known edits with louder competing speech. The existing check
localized **1 of 12** and flagged fewer genuine groups. The candidate
stays outside the app.
[Portable source and range receipt](evaluation-panels/audio-local-context-transfer-2026-11-13.json)

## Frozen method

Earlier pitch and remote-neighbor checks both struggled to place short
edits under overlapping voices. Recent work associates unusual transitions
and some within-phoneme responses with synthesized audio. This test asks
whether the *same samples* near an edit produce different early- and
late-layer representations when the encoder sees a two-second view
instead of the six-second recording. It does not assign phoneme labels,
identify speakers, alter the audio, or calculate a probability of AI use.
[Pitch experiment](audio-overlap-pitch-transfer-2026-11-13.md)
· [Remote-neighbor replication](audio-remote-neighbor-replication-2026-11-12.md)
· [Phonetic attribution research](https://arxiv.org/abs/2607.08586)
· [Adjacent embedding transitions](https://arxiv.org/abs/2604.01083)

One frozen WavLM Base Plus forward sees the original six seconds. A
batch of six overlapping two-second crops sees those same original
samples. Discard frames within 320 ms of a crop edge. At each shared
source position, subtract early-layer cosine distance from late-layer
cosine distance. When two crops cover that position, keep the smaller
disagreement. A reported 800 ms interval compares its median with both
320 ms flanks on an 80 ms grid. The same voice-activity gate must support
at least 60% of the sample positions in the interval and in each flank.
All model inputs derive only from the submitted waveform. The evaluation
uses the model card’s linked CC BY-SA 3.0 terms for the pinned encoder
and the separate MIT speech-activity asset. No model weights or
embeddings are distributed here.
[Pinned encoder](https://huggingface.co/microsoft/wavlm-base-plus/tree/4c66d4806a428f2e922ccfa1a962776e232d487b)
· [Encoder license](https://github.com/microsoft/UniSpeech/blob/main/LICENSE)
· [Existing voice-model terms](commercial-model-audit.md)

The development set is 28 exact same-word deliveries from two
PhonemeDF readers at 0 and −5 dB. There are 24 distinct waveforms.
Twenty ordinary and isolated-background deliveries set the maximum
point, **0.11382773**. The held bank contains 52 distinct six-second
waveforms at −5 dB from four other recorded AMI sessions. Its 12
edits span Chatterbox-generated DuplexGen, ElevenLabs v4, and Kokoro,
four each. Every held edit has ordinary same-reader and other-reader
splices and an untouched counterpart. Four isolated AMI room excerpts
complete the panel. The source recordings previously appeared in other
experiments. ElevenLabs and Kokoro are outside the calibration bank for
this check. They cannot be assumed absent from encoder or incumbent
training. Known edit intervals are evaluation-only.
[Development sources and terms](audio-phoneme-short-edit-intake-2026-11-13.md)
· [Held source qualification and exact room geometry](audio-remote-neighbor-replication-2026-11-12.md)

The report compares the existing point and the candidate's frozen point,
counting a false alarm once per matched source group. To qualify for a
follow-up, the candidate must add at least two useful locations across
two generator families, add at most one false group, and keep all
isolated backgrounds clear. A useful location covers half the edit and
reaches 0.30 temporal intersection over union.

## Held results

| Measure | Existing short voice check | Context disagreement |
| --- | ---: | ---: |
| Useful locations on 12 synthetic edits | 1 | 0 |
| Synthetic flags outside useful edit support | 2 | 4 |
| Ordinary files flagged out of 40 | 7 | 13 |
| Ordinary groups flagged out of 16 | 3 | 5 |
| Isolated room clips flagged out of 4 | 0 | 0 |
| Unassessed files out of 52 | 0 | 0 |
| Useful strongest range before applying the point | 5 | 3 |

The current check found a Chatterbox replacement at **1.680–2.640 s**
within its **1.000–3.000 s** evidence range. The candidate nominated
**2.720–3.520 s** on that same file. Its two Kokoro and one ElevenLabs
useful nominations stayed below the development point. An ElevenLabs
edit at **3.440–4.240 s** produced the candidate's strongest flagged
synthetic response, at **0.640–1.440 s**. Across descriptive budgets of
zero to three held false groups the candidate located no edits. The
existing check found one at two and three false groups. The trial shows
that differences between these two encoder contexts fail to identify
the edited interval under this kind of overlapping speech.

## Local replay

Rebuild the development WAVs with
`scripts.audio_phoneme_short_edit_score.file_inputs`, writing signed
16-bit mono samples at 16 kHz with names `{case}_{level}_{kind}.wav`.
The default held path is the retained remote-neighbor replication
directory. Supply its alternate path with `--held` if needed. Freeze
checks the source hashes, 104 incumbent input-window hashes, prior
receipts, encoder assets and speech-activity asset. The CPU trial
completed 80 full and 80 batch forwards, representing 560 model views.
Replay reopens the native PCM, rechecks source and prior scores and
reconstructs the reported decisions. A second scoring run reproduced
the freeze, development and held receipts byte for byte. The portable standard-library
reader independently checks the fitted point, ranges and false-group
arithmetic from the public scalar receipt.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze fit score replay; do
  backend/.venv/bin/python -B -m scripts.audio_local_context_transfer "$phase" \
    --run "$run" --development-wavs "$development_wavs" || exit 1
done
python3 -B -S backend/scripts/review_audio_local_context_transfer.py \
  docs/evaluation-panels/audio-local-context-transfer-2026-11-13.json
```
