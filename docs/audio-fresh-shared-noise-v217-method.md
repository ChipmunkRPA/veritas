# Fresh three-arm shared-noise preparation

Use the 16 fixed roster positions from each of the guarded v212 genuine Turkish
preparation, v213 fresh Alania preparation and v215 Drifting preparation. Pair
the same ordinal position across the three independently selected rosters;
pairing uses no detector response. Each source must have at least 48,000
production samples. Use exactly its first 48,000 signed-16-bit samples at
16 kHz. A source that is unavailable or too short remains unavailable for all
five of its planned conditions. Other available arms at that ordinal are
retained and mixed together. The set of present arms determines the reference
level, so conditions with a missing arm do not have the same level-matching
context as a complete triplet.

Use channel 1 of the DEMAND kitchen and traffic recordings by Joachim
Thiemann, Nobutaka Ito and Emmanuel Vincent under CC BY-SA 3.0.
[Publisher record](https://zenodo.org/records/1227121). Verify the exact
[admitted asset record](evaluation-panels/audio-demand-kitchen-traffic-assets-v212.json),
publisher archive MD5 and SHA-256, and complete channel PCM SHA-256 before
construction. For ordinal `i`, take both tape excerpts from `4 + 6i` seconds
for exactly 48,000 samples. Every available arm in that triplet receives the
same two source excerpts.

The fixed five conditions are one level- and gain-matched clean control,
kitchen at 0 dB and +5 dB, and traffic at 0 dB and +5 dB. The v216 shared-noise
constructor attenuates foregrounds to the quietest available arm, then applies
one common anti-clipping gain. It quantizes each clean source and each of four
noise residuals separately. Within each tape/level condition, the residual is
byte-identical across all available arms; independently subtract clean from
mixed samples and verify exact equality and the measured SNR within 0.1 dB.
If an available triplet cannot be constructed, retain its affected arms as
unavailable with a reason; do not substitute sources, windows or outcomes.

The expected plan contains 16 triplets × 3 arms × 5 conditions = 240 cases.
Readout reports prepared and unavailable counts per arm and condition, plus
complete-triplet and all three pairwise-coverage counts per condition.
It independently reconstructs and
verifies every saved WAV and private journal row under the existing v108
data-only guard. Source IDs, exact windows, per-case hashes and waveforms remain
private. The aggregate records input preparation and exact sample coverage.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_fresh_shared_noise_v217 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_fresh_shared_noise_v217 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_fresh_shared_noise_v217 readout
```

The source cohorts differ in recording and generation history. The Drifting
cohort shares three stock voices across its 16 planned phrases. The two
background recordings provide a fixed kitchen and traffic comparison. A
later scoring run measures the detector on these voices and recordings.
