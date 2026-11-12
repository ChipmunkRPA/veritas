# Physical-channel agreement under competing speech

The current voice check flagged 16 of 32 genuine mixtures in the paired room study. A rule requiring corroboration in either physical microphone withheld three of those false flags and three correct synthetic flags. Requiring both microphones withheld eight false flags and seven correct synthetic flags. Both rules remain experimental. [Exact aggregate](evaluation-panels/audio-stereo-agreement-2026-11-11.json)

## Comparison

Replay all 101 exact three-second views from the paired stereo study. The two recorded rooms supply matched sequential and simultaneous speech interference at −5 dB mono SNR. The panel contains 32 genuine and 36 synthetic mixtures, 17 matched clean voices, and 16 native room recordings. This panel was already used for the stereo rescue experiment, and its voices have prior project exposure. The present test measures two additional responses per eligible recording using the same pinned classifier and calibration as the mono assessment. It reuses the physical channels' original speech-admission decisions and verifies their binding to the source. [Panel construction and source terms](audio-paired-stereo-overlap-2026-11-11.md)

Compare the two exact physical channel inputs only when the mono direction is AI, both channels pass the same speech-admission rule, and their sample bytes differ. Evaluate two fixed ways to withhold a mono AI result to inconclusive. The first acts only when neither microphone supports AI. The second acts whenever at least one microphone lacks AI support. Neither rule adds an original or AI call. A channel response has the same threshold and class guard as the incumbent; the original-direction stability probe is unnecessary because channel responses contribute only a binary AI support measurement. Advancement requires at least two corrections in each interference condition, at most two correct synthetic calls withheld overall, and zero clean or native flags withheld.

Recent audio detection evaluations report complementary errors across systems and use condition-specific routing. Here both checks use the same model on microphone views of the same scene, which limits the independence of their errors. An earlier spatial-cue trial used different rooms and withheld six synthetic flags after applying a channel check, with no measured false calls on its native genuine panel. This replay measures the trade-off on mixtures with known false calls. [Recent all-type audio benchmark](https://arxiv.org/abs/2608.23437v2) · [Earlier channel corroboration](audio-spatial-overlap-transfer-2026-11-11.md)

## Results

| Condition | Current flags | At least one channel must support AI | Both channels must support AI |
| --- | ---: | ---: | ---: |
| Genuine, sequential interference | 7/16 | 5/16 | 4/16 |
| Genuine, simultaneous interference | 9/16 | 8/16 | 4/16 |
| Synthetic, sequential interference | 12/18 | 10/18 | 9/18 |
| Synthetic, simultaneous interference | 13/18 | 12/18 | 9/18 |
| Clean synthetic voices | 9/9 | 9/9 | 9/9 |

All 41 mixed mono AI calls had two distinct, admitted channels. Nine clean AI calls used identical channel samples and remained unchanged. The eight clean genuine views and 16 native views had no mono AI call; four native views were unassessed. The stricter corroboration requirement withheld correct flags in all three synthetic voice families. Simple physical-channel agreement cannot resolve the background-speech problem on this panel.

The model ran 82 CPU forwards in 28.1 seconds after load and peaked at 2.92 GB process RSS. A separate standard-library replay verified all 101 stereo files, exact mono and channel sample commitments, all 82 channel decisions and both policy outcomes. The aggregate contains condition counts and artifact commitments. The voices, individual responses and waveform files stay local.

## Reproduce locally

Use the original paired stereo evaluation folder, the exact pinned checkpoints and the project Python environment. Run the following from `backend`. The audit reads retained responses and performs no inference.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m scripts.audio_stereo_agreement freeze PAIRED_RUN NEW_RUN
python -m scripts.audio_stereo_agreement run PAIRED_RUN NEW_RUN
python -m scripts.audio_stereo_agreement readout PAIRED_RUN NEW_RUN
python -m scripts.review_audio_stereo_agreement PAIRED_RUN NEW_RUN app/detectors/audio/calibrations/local-speech-v3.json
```
