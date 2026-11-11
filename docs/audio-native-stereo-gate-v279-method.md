# Acoustic channel selection for speech admission

Use all 185 v277 development WAVs across the 37 foregrounds, with five
views per foreground. Bind the completed development receipt, resource
profile, source and window roster, development admission receipt, and exact
private case journal. Verify every native stereo file against its original
hash. Verify the production FFmpeg downmix bytes and clock and both observed
channel hashes before measuring a case. The holdout role stays closed.
[Paired construction](audio-native-stereo-pairs-v277-method.md)

Measure the exact existing 3-second speech gate on the mono downmix and on
each observed channel separately. Use the pinned local Silero checkpoint
and verify the half-window decision against the production window builder.
The worker has a fixed 52,170 frame-forward budget across the 555 measured
views. This stage measures speech coverage with zero authenticity calls.

Define shared activity as the union of the observed channels' speech spans.
Within the three-second source, use complete nonoverlapping 512-sample blocks
wholly in that union as active and blocks wholly outside it as quiet. Require
at least eight of each. Withhold selection if either channel hits a signed
16-bit rail. For each channel, compute `10 log10(active median power / quiet
median power)` with a floor of one squared signed sample on each median.
Choose the higher contrast only when the gap is at least 3 dB. The rule sees
only both exact channels and their activity spans. The known forced noise
assignment enters the analysis after the choice. Repeat the selection after
swapping the two PCM columns and corresponding span sets to verify the same
physical choice, without repeating model inference.

Report mono and selected-channel admission, additions and mono-only coverage,
and the mono-or-selected diagnostic for every condition and origin family.
Report how often the selector chooses the assigned quieter channel on the
forced +5/−10 dB conditions and account for all unselected cases. Report
all 185 swap checks. The paired clips share foregrounds and two recorded
background sessions. The study has no independent population error estimate
and no background-only false-alarm control. A coverage gain would support
a separate classifier comparison. It establishes no authenticity improvement.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_gate_v279 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_gate_v279 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_gate_v279 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_gate_v279 readout
```

Use the existing offline v108 monitor. The readout validates the terminal
receipt and reproduces selection, coverage and accounting from the saved
activity spans and exact channel PCM. Keep the per-case spans and acoustic
measurements private.
