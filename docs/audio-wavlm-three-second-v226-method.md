# Native three-second WavLM feature qualification

Check the feature clock on already consumed three-second audio. Reuse
the [v13 checkpoint and pooling](audio-wavlm-v13-protocol.md), including
its CPU float32 standard loader and exact installed implementation.
The publisher card links the weights to CC BY-SA 3.0. Retain attribution
and share-alike obligations for covered redistribution or adaptation.
The separate code notice is MIT. The original four-second extractor and
its qualification remain unchanged.

Bind the complete guarded [v217 preparation](audio-fresh-shared-noise-v217-method.md)
and [v219 baseline readout](audio-fresh-shared-noise-score-v219-method.md).
From the frozen `traffic_0db` condition, select the first triplet in
source order for which the original speech gate admitted all three
arms. Use one genuine, one Alania, and one Drifting clip. Selection uses
the admission status and original order. It does not use scores or
directions. Verify the three exact saved WAVs and their shared noise
residual. Repeat the genuine input once after all three arms.

Feed the original 48,000 mono samples at 16 kHz, as finite float32 PCM.
Use an all-valid input mask and keep the original input unpadded and
unnormalized. The convolution stack has a stride of 320 samples and a
400-sample receptive field. It produces 149 complete frames. The last
convolution frame consumes `[47360, 47760)` on the original sample clock.
The encoder uses the complete valid feature sequence. Require all 13
hidden states to have the native shape and pool layers 4, 8, and 12 with
the original population means and standard deviations. The result
contains 4,608 finite float32 values. Require the repeated input to
reproduce the selected hidden-state commitments and pooled values exactly.

The study qualifies a feature extraction path. It does not score a head,
fit calibration, compare accuracy across durations or promote the
four-second head to three-second service inputs. Individual source
receipts, feature values, and feature fingerprints stay private. The
aggregate retains counts and whole-artifact commitments.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_wavlm_three_second_v226 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_wavlm_three_second_v226 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_wavlm_three_second_v226 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_wavlm_three_second_v226 readout
```

Preflight verifies inputs and packages without initializing a model.
The four encoder forwards are planned for one owned offline worker
under the unchanged v108 6 GiB neural guard.
