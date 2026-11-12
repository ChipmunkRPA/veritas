# Short voice edits after controlled room-speech attenuation

The existing voice classifier flagged zero of six held synthetic inserts even after idealized 20 dB attenuation of the competing speech. The exact dry foreground produced four flags and three useful locations. This test indicates that materially better waveform reconstruction can still leave the classifier unable to rank short edited and genuine speech reliably. [Results and replay receipt](evaluation-panels/audio-room-reference-transfer-2026-11-12.json)

## Source and fixed comparison

Reuse the 48 exact six-second recordings from the overlapping-window study. Each panel uses six matched genuine and generated insert pairs. Each pair contributes a dry foreground and a mixture with louder competing voices recorded in an AMI meeting. Selection and confirmation share no declared foreground group or room. Both panels and all generator families had earlier project exposure. The 0.8-second edited ranges stay fixed. In each file, score the two previously admitted windows at 1–3 and 3–5 seconds. Holding these windows fixed isolates waveform changes from speech admission and timestamp choice. [Earlier source, construction and rights](audio-overlap-window-transfer-2026-11-12.md) · [AMI corpus license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The competing waveform comes from the difference between the exact gain-matched dry and mixed PCM, with up to one integer sample unit of rounding uncertainty. Compare three transformed mixtures with the original mixture and exact dry endpoints.

- **Noncoincident room reference.** Use mean competing-speech power from the four seconds outside the scored window. This approximates a separate background-only sample from the same scene.
- **Aligned room reference.** Use the known background-only time-frequency power within the window. Both spectral views apply the same square-root power subtraction, retaining the mixture phase and a 20% magnitude floor. Analysis uses a 512-sample periodic Hann window with a 128-sample hop. [Time-frequency masking research](https://research.google/pubs/on-training-targets-for-supervised-speech-separation/)
- **Ideal residual.** Subtract 90% of the known competing waveform amplitude from the exact mixture. This leaves a controlled 20 dB reduction of the background without spectral masking artifacts.

The original and dry views retain the earlier maximum-genuine operating point. For each new view, fix the strict maximum across its six processed and six dry genuine selection controls. Use the higher of the two window scores, and require at least 0.30 intersection over union and 0.50 recall to call its location useful. A spectral arm qualifies for a further real separate-channel trial only with at least two useful held locations from two generator families, at most one genuine false flag and four of six correctly ordered genuine/edited pairs. The published checkpoint audit covers the existing local weights. [Checkpoint terms](commercial-model-audit.md)

## Held outcomes

| Input to the same voice classifier | Median reconstruction SNR | AI edits flagged /6 | Useful locations /6 | Genuine false flags /6 | Edited score above matched genuine /6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Observed mixture | −5.16 dB | 0 | 0 | 0 | 2 |
| Noncoincident room reference | −3.64 dB | 0 | 0 | 0 | 2 |
| Aligned room reference | +5.10 dB | 0 | 0 | 0 | 2 |
| Ideal residual | +14.84 dB | 0 | 0 | 0 | 2 |
| Exact dry foreground | — | 4 | 3 | 1 | 4 |

Reconstruction SNR compares each transformed integer waveform directly with its gain-matched dry reference. Across both held rooms, every synthetic result from the observed and attenuated mixtures remained inconclusive. Even without any decision point, the best window overlapped the edit usefully in just two of six ideal-residual cases, compared with five of six dry cases. The poor matched-pair ordering gives a second measure of the failure, independent of the operating point.

Neither reference passed the frozen follow-up requirements. The two spectral references and ideal residual have privileged background information unavailable in ordinary mono uploads. The result supports testing a model explicitly trained or calibrated on short voice inserts with residual competing speech, and measuring localization alongside the quality of any upstream separation. This test supplies no new confidence or customer-facing finding. [Recent mixed-authenticity detection and source separation research](https://arxiv.org/abs/2609.03620)

## Verification and replay

The two CPU runs completed 144 new classifier forwards in 80.2 seconds, with a peak process RSS of 2.90 GB. The independent reviewer rebuilt the transformed waveforms using a separate NumPy FFT and overlap-add implementation. It verified all 240 window hashes, reproduced the five selection points and ten role-by-arm decision cells, and found no clipping. It rejected a changed window hash even when the outer ledger checksum was updated. It used saved model scores without repeating inference.

Use the completed and independently audited earlier overlap-window run as `parent`, the exact local checkpoint directory as `weights`, and a fresh private directory as `run`.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
args=("$run" "$parent" "$weights")
backend/.venv/bin/python -B -m scripts.audio_room_reference_transfer freeze "${args[@]}"
backend/.venv/bin/python -B -m scripts.audio_room_reference_transfer score "${args[@]}" --panel selection
backend/.venv/bin/python -B -m scripts.audio_room_reference_transfer fit "${args[@]}"
backend/.venv/bin/python -B -m scripts.audio_room_reference_transfer score "${args[@]}" --panel confirmation
backend/.venv/bin/python -B -m scripts.audio_room_reference_transfer report "${args[@]}"
backend/.venv/bin/python -B -m scripts.review_audio_room_reference_transfer "${args[@]}"
```
