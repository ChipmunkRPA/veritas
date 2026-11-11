# Transfer test for speech admission in recorded noise

Reuse the fixed speech-admission rule from v256. The current gate requires half of a three-second window to fall in returned speech spans. The fallback admits the same complete waveform when the activity model finds a continuous returned span of at least one second. The production window builder independently checks the current-gate decision for each waveform. The experiment runs the local speech activity model only. [Earlier admission comparison](audio-noise-speech-gate-v256-method.md)

Use the 16 planned positions from each of the verified v217 VoxForge, Alania and Drifting source cohorts. The prepared foregrounds differ from the RUEG, later Alania and Okur sources in v256. Check exact waveform and prefix fingerprints against the consumed v248 panel. Check catalog voice IDs for Alania separately because both studies use that synthesis family. Drifting utterances share three publisher stock voices. The known source groups are distinct from the development panel, and the v217 foregrounds have prior exposure in different mixtures. They provide a transfer comparison without claiming previously unseen foregrounds across the whole project. [Source preparation](audio-fresh-shared-noise-v217-method.md) · [Development panel](audio-matched-background-panel-v248-method.md)

Verify the exact DEMAND kitchen and traffic recordings from Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent under CC BY-SA 3.0. Assign three-second excerpts at `99 + 6 × ordinal` seconds within each tape. This fixed grid is temporally disjoint from the earlier v217 grid. Preserve matched clean, kitchen 0 and +5 dB, and traffic 0 and +5 dB. Use the inherited level-matching, anti-clipping and shared quantized-residual construction. Background-only controls use every complete three-second interval on the later grid outside the planned mixture intervals. Reject repeated waveform fingerprints against the former tape cuts. Choose all intervals by their coordinates before inference. The two environmental recordings have no per-excerpt listening labels, so an admission on a background-only control cannot establish a speech false alarm. [Publisher record and attribution](https://zenodo.org/records/1227121) · [Verified asset receipt](evaluation-panels/audio-demand-kitchen-traffic-assets-v212.json)

Freeze exact source and preparation commitments. The v108 guard starts an offline worker with the same activity checkpoint, coverage rule and serving window check as v256. Reconstruct the aggregate from the resource and private observation receipts. Keep exact input identities, source coordinates, spans and waveforms private. Report planned and prepared variants, baseline and fallback admissions, and source-group counts for newly admitted speech. Keep recordings visible in the background denominator. This study does not classify authenticity, fit a policy, or change serving.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_transfer_v267 preflight
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_transfer_v267 freeze
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_transfer_v267 run
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_transfer_v267 readout
```
