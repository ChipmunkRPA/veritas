# Authenticity after stereo speech admission

Use all seven v279 development cases rescued by the fixed acoustic channel
selector. Include every forced-noise case in the same genuine and expressive
families where the mono downmix and selected channel both passed the speech
gate. This adds 26 controls. The full comparison includes 13 genuine and
20 expressive noisy cases across the river and washroom backgrounds.
Membership follows saved speech activity and channel choice before any
authenticity score. The equal-gain mixtures produced no added speech admission
in v279. [Acoustic channel selection method](audio-native-stereo-gate-v279-method.md)

Score the exact production mono downmix and the exact selected native
channel on each noisy case. Verify the v277 PCM and source clock commitments.
Link each noisy case to its own gain-matched clean recording with the same
source slot, voice group, and saved speech eligibility. Score each of the 18
unique matched clean waveforms once, so the study runs 84 waveforms total.
Preserve both tape rows when they reuse the same clean score. Report the
unique clean results separately from the paired view counts.

Use the pinned v261 classifier, calibration, stability probe, and loader
configuration in the previously verified CPU runtime. The fixed budget is
168 classifier forwards. The stage reuses saved v279 activity and loads no
other scoring model. Report a score from gate-rejected mono only as a
counterfactual. Its corresponding gate-respecting result remains not assessed.
The selected-channel score describes a short development window. Reused
foregrounds, shared background sessions, and forced noise imbalance limit
transfer claims. The holdout role stays closed and the experiment has no
product confidence effect. [Earlier downstream classifier method](audio-noise-speech-rescue-downstream-v270-method.md)

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_authenticity_v282 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_authenticity_v282 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_authenticity_v282 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_authenticity_v282 readout
```

Use the existing offline resource guard. Keep exact source links, PCM hashes,
and classifier scores in private receipts. Publish only aggregate direction
counts and artifact commitments after reviewing the terminal result.
