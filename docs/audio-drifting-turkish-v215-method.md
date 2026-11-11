# Independent Turkish synthetic source preparation

The v215 panel prepares 16 newly authored Turkish phrases with the released
Drifting TTS v3.1 pipeline. The publisher licenses the ONNX weights under
CC BY 4.0 and the inference code under MIT. The v3.1 BigVGAN-v2 vocoder is
derived from NVIDIA’s MIT release. Drifting TTS uses a mel spectrogram and
a different decoder from the existing VoxCPM2 Alania panel. Use the three
publisher stock voices in fixed rotation with a fixed per-phrase seed.
Voice profiles and prompts remain unchanged throughout generation. No
speaker reference audio is supplied. Stock voice provenance beyond the
publisher records is unverified. [Pinned model](https://huggingface.co/Vyvo/drifting-tts-tr/tree/0906c46992410a2032c6f5777b887f8f1780a6fd) · [Pinned upstream code and license](https://github.com/kadirnar/drifting-tts/tree/b8cbe55d1614e2c79f3e7f997985c3144f1fbb4d) · [NVIDIA BigVGAN-v2](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x)

The local source directory contains the pinned publisher metadata, license,
and exact upstream orchestration and export source for audit. Bind all
three publisher ONNX files with fp16-stored weights, including the fp16
text encoder. The exporter casts the stored weights to fp32 in the graph.
Use the isolated ONNX Runtime 1.23.2 package on CPU with one intra-op and
one inter-op thread. The application environment has no new dependency.
Bind the installed package files, Python and NumPy versions, and the
production FFmpeg and FFprobe binaries before starting the worker.
[Upstream MIT notice](../backend/scripts/licenses/Drifting-TTS-MIT.txt)

Use the pinned browser frontend’s character IDs and blank interleaving
for the fixed plain Turkish text. Apply each voice’s publisher duration
scale and repeat each token’s condition for its rounded frame count.
Generate the same seeded uniform and normal noise stream as the publisher
browser method, with temperature 0.3, guidance 2.0, and unit length scale.
Run one encoder, one acoustic generator, and one whole-sentence vocoder
pass for each admitted input. Reject a predicted duration outside three
to four seconds before the acoustic generator. Preserve every unavailable
slot. No text substitution, duration tuning, padding or crop is applied.

Preserve the complete native 24 kHz float32 waveform and run its exact
bytes through the production decoder to 16 kHz mono signed-16-bit PCM.
Verify the continuous source timeline, exact sample extent, and WAV
round-trip. Keep per-case audio, authored texts, seeds and hash receipts
private. The published result contains aggregate coverage, call counts,
resource bounds and commitments. The existing neural resource guard
monitors generation with a 6 GiB initial RAM requirement. The worker
does no authenticity scoring, fitting or holdout access.

The public comparison should keep generator family and stock voice
dependence visible. These 16 sources share three stock profiles and
provide a bounded synthetic challenge for the matched acoustic study.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_drifting_turkish_v215 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_drifting_turkish_v215 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_drifting_turkish_v215 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_drifting_turkish_v215 readout
```
