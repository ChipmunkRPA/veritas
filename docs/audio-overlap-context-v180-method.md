# Measure competing-speech context on the PACC-P development replay

Use the public full-precision Sherpa ONNX export of pyannote speaker
segmentation 3.0. The release includes the original MIT license and the
export's timing metadata. No account-gated checkpoint, speaker identity
classifier, waveform separator or new voice-origin model is required.
Verify the archive length and contents, the published Apple Silicon ONNX
Runtime wheel and the isolated runtime files before running.
[Upstream model and license](https://huggingface.co/pyannote/segmentation-3.0) · [Sherpa export and source](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models) · [ONNX Runtime 1.23.2](https://pypi.org/project/onnxruntime/1.23.2/)

Replay all 98 exact genuine WAVs from the previously observed PACC-P
sample and bind the prior serving decisions. The candidate makes one
CPU forward per case. Right-pad with digital silence to the model's
documented 160,000-sample input as in the release example. Keep only
frame outputs whose complete 991-sample convolutional receptive field
lies inside the three-second waveform. Frame stride is 270 samples at
16 kHz. A frame's most likely class can describe no speaker, one speaker,
or two simultaneous speakers. Record the summed two-speaker mass as
a continuous observation. A descriptive file-level marker uses at least
250 ms of contiguous frames whose most likely class has two speakers.
Frame cells are centered on the native receptive fields. The model
also sees the padding through its recurrent layers.

Report the marker's overlap with existing speech-route directions by
noise condition and recording source. The source treatment documents
babble, environmental noise and instrumental music. It does not supply
frame-level overlap labels for these prefixes. The marker measures
perceived simultaneous speech and gives no speaker identity, synthetic
origin or authorship confidence. This is retrospective engineering
evidence. A separate matched genuine and synthetic experiment is needed
before considering a serving policy.
[PACC-P source and prior scoring](audio-pacc-false-alarm-v177-result.md) · [Related component-separation research](https://arxiv.org/html/2610.17617v1)

Install the verified wheel into `site` under the local runtime directory,
with the exact wheel alongside it. Keep the publisher archive in the local
study archive directory. Then run the bounded offline worker.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_overlap_context_v180 freeze /path/to/runtime
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_overlap_context_v180 run /path/to/runtime
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_overlap_context_v180 readout /path/to/runtime
```

Retain the intervals and per-input observations privately. Publish counts,
medians and whole-artifact commitments.
