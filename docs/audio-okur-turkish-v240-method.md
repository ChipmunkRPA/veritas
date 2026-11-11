# Independent Okur stock-voice source proof

Use the three native Okur v1 ONNX graphs, with publisher CC BY 4.0 weights and Apache 2.0 code. The released speaker is an anonymous Common Voice contributor. The publisher trained the model from scratch on Turkish speech using ACE-Step 1.5 latent targets and an independently trained HiFi-GAN-style decoder. That lineage supplies a separate generator from Alania's VoxCPM2 pipeline. Keep the Okur stock voice in one group and use a separate genuine-speech corpus for the comparison. [Pinned weights, notice and source information](https://huggingface.co/ahakan/okur-tts/tree/0c1ae1a9c5e7bcaa46bdff732bc7182c9ae7bd70) · [Pinned source and Apache license](https://github.com/ahmethakanbesel/okur-tts/tree/0889eda23a3eaeda33f8c0af3af729261907bcab)

Freeze eight new plain Turkish sentences already in the published alphabet. For this proof the sentences enter at the documented normalized-letter interface. Use the published word ownership and cumulative half-even duration rule, the browser's seeded Mulberry32 and Box–Muller noise stream, and its four fixed sampler steps at speed and temperature one. Bind the exact native opset-18 graphs and the local v215 ONNX Runtime 1.23.2 package. This isolated proof uses no package installation. Compatibility with these actual graphs remains to be verified by a guarded load, as the publisher's full Python package declares a newer runtime. [Publisher export and reference inference](https://github.com/ahmethakanbesel/okur-tts/blob/0889eda23a3eaeda33f8c0af3af729261907bcab/src/okur/export_web.py)

Run acquisition separately from generation. Ten pinned asset paths on the two first-party repositories bind exact bytes, with three weights totaling 23,204,782 bytes. The first-party graph redirects were checked using response headers only. The downloader permits only the observed HTTPS Hugging Face Xet CDN pattern after matching the pinned revision, size and SHA-256, with no second redirect or automatic retry. The acquisition stops at the first error and retains any partial bytes without marking the asset verified. The guarded worker reads only verified local assets and uses one thread per ORT execution pool. The app's normal runtime stays outside this experiment.

Use the 3–4 second native duration interval already used for earlier Turkish source qualification. A text-stage prediction outside 75–100 frames stays unavailable with its one graph call recorded. Each admitted waveform stays at its complete native 48 kHz extent, then passes through the production decoder for 16 kHz sample-clock and WAV round-trip checks. Replay the first qualified phrase exactly once and require identical native and production bytes. Publish aggregate outcomes, the whole-artifact commitments and the sampled resource profile. This stage makes no detector calls and fits no confidence calibration.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.audio_okur_turkish_v240 preflight
backend/.venv/bin/python -B -m scripts.audio_okur_turkish_v240 freeze
backend/.venv/bin/python -B -m scripts.audio_okur_turkish_v240 acquire
backend/.venv/bin/python -B -m scripts.audio_okur_turkish_v240 run
backend/.venv/bin/python -B -m scripts.audio_okur_turkish_v240 readout
```
