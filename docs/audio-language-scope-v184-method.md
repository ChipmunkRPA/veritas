# Bounded language-scope diagnostic

This diagnostic compares the 24 already qualified full Turkish synthetic
Alania v179 clips with the 14 qualified genuine English LibriSpeech development
clips in v169c. The English clips contain seven distinct source groups and
voices, not 14 independent speakers. Turkish has 24 designed voice/text groups,
all from one synthetic generator and the publisher's clean post-filtered
condition. Language, corpus and real-versus-synthetic origin are confounded;
differences cannot be attributed to language alone. No v169c holdout is read.

The inputs retain their exact prepared 16 kHz mono PCM commitments: complete
2.5–4 second Turkish files and the first three seconds of English files.
Verify the v179 guarded preparation and v169c development manifest and each
selected WAV and PCM hash. Missing or corrupt committed media stop the run;
inference errors on otherwise valid media are retained as unavailable. Do not
replace, crop or select a case based on the result.

Use the locally installed [multilingual Whisper Tiny checkpoint](https://huggingface.co/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af)
and preprocessing configuration pinned in v110 and the generation configuration
and tokenizer assets pinned in v163. The publisher checkpoint declares
Apache-2.0; the [Whisper source license](https://github.com/openai/whisper/blob/main/LICENSE)
is MIT. The model and token-table assets are checked by hash.
The English-forced transcript procedure in v163 is not used.

CPU float32, deterministic algorithms, one thread: preprocess each entire clip
with the frozen v110 30-second padded log-Mel settings. Call the native full
encoder/decoder model once with only the start-of-transcript token. Restrict the
first-position logits to **all** publisher language tokens and normalize them;
there is no forced language, transcription, generation, prompt, authenticity
score, or tuning. Report the top language as English, Turkish or other. Report
top-two probability margin in the fixed descriptive bins `[0,.1)`, `[.1,.25)`,
`[.25,.5)`, `[.5,1]`; these margins are uncalibrated model responses and are not
service acceptance cutoffs. Report unavailable counts in the planned
denominator and English clip and grouped counts separately.

This is a small development-only check of whether a language identification
signal might help flag inputs outside an English reference. It does not validate
a language or authenticity policy, establish calibration, determine authorship,
measure unseen language or generator performance, or justify changing serving
behavior. No model parameters, thresholds or serving defaults change.

From the repository root, with the pinned files and prior receipts installed,
the owner may run these explicit stages sequentially under the existing v108
CPU resource guard. Each preserves attempted output and refuses overwrite.
Input identities, source paths and per-case responses remain private; the
readout contains aggregates and commitments only.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_language_scope_v184 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_language_scope_v184 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_language_scope_v184 readout
```

The model-free unit tests need neither the audio inputs nor model inference:

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_audio_language_scope_v184.py
```
