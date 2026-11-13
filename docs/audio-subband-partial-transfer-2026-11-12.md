# Separate frequency views for short edits in noisy speech

The [current overlapping-window review](audio-focused-window-transfer-2026-11-12.md)
localized three of 18 edits under loud rain and zero of 18 under a louder
train recording. This comparison tests whether scoring an extra view with
part of the spectrum attenuated finds a missed edit. Recent mixture research
found that different front ends rely on different bands. That motivates
testing the views separately, with the speech and background controls kept
in the decision. No paper code or checkpoint is used. [Frequency probing research](https://arxiv.org/html/2610.17617v1) · [Spectral residual research](https://proceedings.mlr.press/v306/hidekel26a.html) · [Installed model terms](commercial-model-audit.md)

## Fixed procedure

Run a fourth-order Butterworth band-stop filter in both directions across
the exact native recording, preserving the sample count. Compare removing
80–450 Hz with removing 2,200–4,000 Hz. A filtered three-second input can
add evidence only where the native reading was inconclusive or unassessed.
Require the usual minimum speech activity in the filtered window. For each
view, freeze the largest admitted genuine or background-only logit from
selection. Apply the point strictly. Select the common 1.5 seconds when two
adjacent native or same-view responses agree. Preserve the existing ranges.
The two frequency views are correlated measurements of the same recording.

Selection uses 25 exact deliveries from the earlier staggered study, with
seven genuine speaker groups outside the 30 held groups. It includes four
isolated excerpts of restaurant and coastal sound. Held uses the 98 exact
recordings from the narrower-window transfer. It includes clean speech,
rain and train mixtures at −5 dB, and isolated background. Those recordings
were already measured by the original voice check. They were outside the
new filter point selection. The original source and license details are
recorded in the linked studies. [Selection sources](audio-noise-window-transfer-2026-11-12.md) · [Held sources](audio-focused-window-transfer-2026-11-12.md)

The frozen gate requires four new noisy groups across three generator
families and both scenes. It allows up to two new genuine groups and three
additional percentage points of marked genuine time in noisy speech, with
no new clean genuine or isolated flags.

## Results

| Known edits with a useful location | Current | With low-band stop | With high-band stop | Both views |
| --- | ---: | ---: | ---: | ---: |
| Gain-only | 13/18 | 15/18 | 14/18 | 16/18 |
| Rain | 3/18 | 3/18 | 3/18 | 3/18 |
| Train | 0/18 | 0/18 | 0/18 | 0/18 |

Neither view added a flag on the 36 genuine held deliveries or eight
isolated excerpts. The three gain-only locations cost an extra 5.40
seconds of highlighted genuine speech inside edited files. No noisy
location or marked genuine time changed. The study missed all three
noisy gain requirements. Keep the existing detector. The gain-only
finding would need new source speakers, codecs and recording conditions
before it can support a separate product check.

The native reader reconstructed all 572 filtered window commitments.
It also recalculated the two separate points from 15 and 20 admitted
selection responses and replayed all held decisions. The public ledger
contains measurements and hashes, with audio retained locally.
[Frozen result](evaluation-panels/audio-subband-partial-transfer-2026-11-12.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_subband_partial_transfer review \
  docs/evaluation-panels/audio-subband-partial-transfer-2026-11-12.json
```

To rerun inference, prepare `selection` and `held` using the linked studies.
Use a new `run` directory and the existing pinned voice assets.

```sh
export PYTHONPATH=backend VERITAS_AUDIO_DETECTOR_MODE=candidate VERITAS_BILLING_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_subband_partial_transfer run "$run" \
  --selection "$selection" --held "$held"
backend/.venv/bin/python -B -m scripts.audio_subband_partial_transfer review "$run/receipt.json" \
  --selection "$selection" --held "$held"
```
