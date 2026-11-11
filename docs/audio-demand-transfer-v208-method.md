# Speech detector transfer under recorded background noise

Use the complete verified v204 journal. Retain all 24 planned genuine
slots, all 24 synthetic slots, and all six conditions. Run the fixed
English-reference voice route used in the earlier Turkish studies.
Freeze model and asset identities before scoring. The original-gain
clean conditions must reproduce the previous directions, window
coverage, stability disposition and numeric scores within 0.000001.
[Noise preparation](audio-demand-controls-v204-method.md) · [Prior genuine run](audio-voxforge-tr-transfer-v201-result.md) · [Prior synthetic run](audio-alania-transfer-v179-result.md)

Use a separate bounded worker for each condition. Record actual model
forwards, the speech gate disposition, direction and case coverage.
Compare each noisy input to its own gain-matched clean counterpart.
Compare the gain-matched clean input to its original-gain clean
counterpart. Report the direction transitions, false AI calls on
genuine inputs, missed AI on synthetic inputs, and unavailable and
inconclusive counts. A speech gate rejection remains unavailable.

This is a paired perturbation experiment across two actual background
recordings. The genuine and synthetic sources differ in their recording
and production histories. Keep absolute error counts and paired changes
separate. The experiment does not fit a model, adjust calibration, or
establish accuracy under natural speech overlap.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_transfer_v208 freeze
for batch in 0 1 2 3 4 5; do
  PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_transfer_v208 run "$batch"
done
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_transfer_v208 readout
```

Use the unchanged v108 900-second neural guard for each worker. Keep
waveforms, exact source ranges and individual scores private. The public
readout contains only aggregate counts and whole-artifact commitments.
