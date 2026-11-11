# Twenty-four additional Okur stock-voice source slots

Use the verified v240 graph assets, CPU runtime and guarded production-waveform preparation. The prior guarded proof qualified four sentences with 50–56 characters and seven or eight words. Author 24 distinct new normalized sentences within that observed length range before running any graph. Keep speed, sampler steps, graph code and the fixed three-to-four-second duration interval. Length is a prospective text choice. The actual duration check still determines which slots qualify. [Verified v240 proof](audio-okur-turkish-v240-result.md) · [Publisher weights and license](https://huggingface.co/ahakan/okur-tts/tree/0c1ae1a9c5e7bcaa46bdff732bc7182c9ae7bd70)

Verify the earlier published result against its unchanged freeze, asset acquisition, private case and replay journals, and monitored profile. Freeze the new plan in a separate workspace. Exclude previous texts, exact letter encodings and seeds. Compare the resulting native and prepared fingerprints against the prior qualified waveforms. Keep all duration-unavailable slots and repeat the first qualified utterance exactly once. The worker reuses the existing local assets and isolated runtime offline.

All qualifying utterances share one stock voice and one generator. They are suitable as a separate held-out generator challenge when the rest of the benchmark is ready. This preparation makes no authenticity detector calls and fits no confidence calibration.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.audio_okur_turkish_followup_v245 preflight
backend/.venv/bin/python -B -m scripts.audio_okur_turkish_followup_v245 freeze
backend/.venv/bin/python -B -m scripts.audio_okur_turkish_followup_v245 run
backend/.venv/bin/python -B -m scripts.audio_okur_turkish_followup_v245 readout
```
