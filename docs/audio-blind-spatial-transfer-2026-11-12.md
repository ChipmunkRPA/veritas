# Stereo separation before screening short voice edits

On six previously scored edits delivered with loud competing speech, a blind stereo separator improved localization from **zero to two useful ranges** at separately frozen points. Both arms flagged two genuine source groups, with five ordinary recordings flagged after separation versus four in the original arm. At zero or one false group, neither arm localized an edit. The result supports a separate room test. [Aggregate receipt](evaluation-panels/audio-blind-spatial-transfer-2026-11-12.json)

## Fixed comparison

The input panel contains 52 six-second stereo deliveries from the [measured-room study](audio-measured-spatial-transfer-2026-11-12.md). Selection uses a Surrey impulse-response recording and two AMI sessions. The held panel uses BBC Research & Development measurements and two other AMI sessions. Each role has six short AI voice edits from three generator families and 20 genuine controls. Every edit has same-reader, other-reader and unedited controls from the same foreground group. The competing feed is 5 dB louder across the unedited host and includes overlapping speech. Both roles had already been scored in the measured-room study. [Impulse response and recording terms](audio-measured-spatial-transfer-2026-11-12.md#sources-and-fixed-comparison)

The two-channel implementation uses iterative-projection independent vector analysis and projects each estimate back onto the stereo mean. Each estimate has the original sample count. Their sum matches the submitted stereo mean before quantization. The routing rule selects the estimate with less left-to-right imbalance relative to its own mono energy. A poor spatial rank or less than 1 dB of separation between the two imbalance measurements leaves the pair unassigned. The calculation uses only the submitted channels. The test's known stems are used to measure reconstruction error and establish the best of the two possible candidates. No third-party separator implementation or weights are used. [Original update rule](https://doi.org/10.1109/ASPAA.2011.6082320) · [Back-projected demixing](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000922.pdf) · [Component-level detection research](https://arxiv.org/abs/2606.10791)

| Median change from the stereo mean on six AI edits | Surrey selection | BBC held |
| --- | ---: | ---: |
| Blind selection, whole clip | +5.90 dB | +5.45 dB |
| Blind selection, inside edited interval | +6.27 dB | +5.21 dB |
| Measured-room cancellation, inside edited interval | +8.67 dB | +9.81 dB |

The fixed blind rule selected the same component as the exact-foreground comparison on all 12 edited inputs. It also selected a component for all four background-only controls, so selection alone cannot establish foreground presence. The two synthetic positions have known, stationary paths in each room. Arbitrary stereo placement, moving subjects and multiple competing playback paths need a separate check.

## Voice screening

The same pinned voice classifier scores 1–3 and 3–5 seconds. The comparison keeps a time window only when the speech gate admits it on both inputs. Each arm's decision point is the maximum genuine response from the eight selection groups. A useful range must cover half of the edit and reach 0.30 temporal intersection over union. Both source groups and generator families must differ for added locations to qualify. A change qualifies for fresh testing with at least two net additional useful held locations, at most one additional false genuine group and no new background-only flag. [Checkpoint and speech-gate rights](audio-pacc-portable-replay.md#inputs-and-rights)

| Held outcome | Original stereo mean | Blind selected component |
| --- | ---: | ---: |
| AI recordings flagged / 6 | 1 | 3 |
| Useful edit locations / 6 | 0 | 2 |
| Ordinary recordings flagged / 20 | 4 | 5 |
| Genuine source groups flagged / 8 | 2 | 2 |
| Background-only flags / 2 | 0 | 0 |
| Unassessed / 26 | 0 | 0 |

One new useful range is a clone at 3–5 seconds, with the exact edit at 3.84–4.80 seconds. The other is an expressive voice at 1–3 seconds, with the edit at 1.84–2.80 seconds. Both reach 0.48 overlap. Neither method found a useful location at retrospective budgets of zero or one false genuine group. At two groups the result is again two versus zero. The earlier historical point on the original input gave zero AI and zero genuine flags here. This reused panel supports testing on a new room and different speakers, and the detector and overall confidence remain unchanged.

## Local reproduction

Use the [audited parent run](audio-measured-spatial-transfer-2026-11-12.md#replay) and a new private run directory. The acoustic replay uses NumPy FFT in place of SciPy FFT and reproduced the selected component, exact quantized output commitments and reconstruction summaries. The voice audit reopens the original and separated sample windows, verifies the hashes and recalculates decisions from saved responses. Two separate model launches reproduced identical saved responses and admission decisions. The scored run made 201 classifier forwards. Source recordings and raw model scores stay private.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze run replay; do
  backend/.venv/bin/python -B -m scripts.audio_blind_spatial_transfer "$phase" "$parent" "$run" || exit 1
done
for phase in freeze score audit; do
  backend/.venv/bin/python -B -m scripts.audio_blind_spatial_voice "$phase" "$parent" "$run" || exit 1
done
```
