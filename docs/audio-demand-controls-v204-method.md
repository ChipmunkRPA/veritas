# Paired Turkish speech with recorded environmental noise

Use all 24 planned VoxForge contributor slots and all 24 previously
prepared Alania synthetic clips. The genuine set has 19 prepared
recordings and five unavailable slots. Preserve their exact signed
production samples and original lengths. The two foreground sources
have different recording, production and preparation histories.
[Genuine results](audio-voxforge-tr-transfer-v201-result.md) · [Synthetic results](audio-alania-transfer-v179-result.md)

Use the publisher's washroom and river recordings from DEMAND 1.0,
by Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent. The deposit
lists the recordings under CC BY-SA 3.0. Verify the publisher's
archive checksums and bind the exact downloaded bytes. Read only
the first microphone channel from each 16 kHz file. Allocate a
separate window for every planned foreground, starting at 4 seconds
and advancing by 6 seconds in fixed roster order. The excerpts
represent two recorded locations and can contain other sounds.
[DEMAND data and attribution](https://zenodo.org/records/1227121)

Each foreground has six planned conditions. Preserve the source
clean signal. Make a gain-matched clean control and four noisy
versions from the two environments at 0 and +5 dB whole-window
SNR. Use one common attenuation across the two environments and
levels for each foreground. Quantize clean and background separately.
Recompute the exact signed sample residual and measure SNR after
quantization. Verify the saved waveform byte for byte. Keep each
unavailable foreground in all six condition denominators.

The data-only run uses the existing v108 guard. The readout reconstructs
the complete private journal and waveforms. Keep all source audio and
mixes private. The next detector run can measure direction and speech
gate coverage within each matched source and relative to both clean
controls. These two tapes cannot establish noise-population accuracy
or performance under competing talkers. The current preparation
produces no model scores or calibration changes.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_controls_v204 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_controls_v204 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_controls_v204 readout
```
