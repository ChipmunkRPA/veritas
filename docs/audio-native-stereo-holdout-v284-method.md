# Held-out stereo speech evaluation

Use all 200 prepared v277 holdout cases from 40 exact foreground recordings.
Keep the existing roles, families, connected groups, and five treatments for
each foreground. The panel includes 16 genuine, six clone, 12 expressive, and
six stock recordings. The holdout backgrounds are the two distinct DEMAND
kitchen and traffic sessions. Prior project studies used the same sessions'
first channel. This panel uses the bound ch02/ch16 pairs and controlled mixes.
The v284 freeze binds the complete case journal and earlier verification while
keeping the actual WAVs closed. [Stereo preparation](audio-native-stereo-pairs-v277-method.md)

Run the unchanged v279 speech gate and acoustic channel choice on the exact
production mono and observed stereo channels. Verify each case against its
native waveform, channel hashes and FFmpeg mono clock. Use the same union of
activity spans, block support, rail check and 3 dB contrast gap, and repeat
selection with the two columns swapped. All 600 views and 56,400 speech-activity
frame forwards are accounted for. [Acoustic rule](audio-native-stereo-gate-v279-method.md)

After independently replaying that gate stage, seal every gate-admitted
mono view and every gate-admitted selected-channel view. Score each through
the unchanged v261/v282 CPU classifier, calibration, and stability probe.
The maximum is 400 views and 800 classifier forwards. Views without gate
admission remain unassessed. Each of the 200 cases still receives one
disposition per route. The stage carries the measured waveform scores only.
[Development classifier comparison](audio-native-stereo-authenticity-v282-method.md)

The primary comparison keeps a mono AI or original direction and admits a
selected-channel AI direction only where mono is inconclusive or unassessed.
Without that AI direction, the mono disposition persists. The optional
channel-priority diagnostic uses the same selected view whenever its gate
admits it, with mono as the fallback. This uses the same classifier receipts.
The policy and development replay were locked before this held-out study.
[Fixed policy replay](audio-ai-only-rescue-v283-method.md)

The report independently reconstructs all decisions from the terminal
score and activity receipts. It gives correct AI and original directions,
genuine falsely leaning AI, synthetic falsely leaning original, synthetic
without an AI direction, inconclusive, and unassessed counts. Keep totals by
family, condition, and treatment. Report additional AI directions and the
number of connected groups with added correct and wrong AI calls. Distinguish
clean, equal-gain, and forced +5/−10 dB cases. Five treatments share each
foreground and multiple cases share the same background session. Inferences
remain limited to this controlled sample and actual stereo speech capture
still needs separate validation. Results carry no serving or confidence change.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_holdout_v284 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_holdout_v284 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_holdout_v284 run-gate
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_holdout_v284 gate-readout
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_holdout_v284 seal-classifier
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_holdout_v284 run-classifier
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_holdout_v284 readout
```

Use the unchanged offline v108 resource guard for both workers. Keep source
links, waveforms, raw scores and the classifier view plan private. The aggregate
report contains counts, measured forwards and artifact commitments.
