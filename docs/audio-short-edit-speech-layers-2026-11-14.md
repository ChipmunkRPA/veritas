# Short voice edits under competing speech

This trial tests whether a pretrained speech encoder gives a short-edit detector
more useful evidence when louder voices are present. The same six-second
recording goes into each candidate as submitted. The caller supplies one file.

## What changed

The prior training study used log-Mel spectra. This comparison replaces those
inputs with frame-level states from three frozen speech-encoder layers. The
encoder reads original 16 kHz PCM. Its frames have 20 ms spacing, with the first
center at sample 200. A fixed signed projection reduces the three states to 64
values at each center. Interpolation aligns them to the existing head grid.
The model architecture, loss, source splits, speech admission and 0.50, 0.75 and
1.00 second proposals stay fixed. The head sees both short-edit content and
its boundaries during training. This follows recent work that evaluates the
two signals together, with a local implementation and existing source labels.
[Boundary and Intra-Segment Learning for Partial Audio Deepfake Localization](https://arxiv.org/abs/2609.25822)
· [Split and Conquer Partial Deepfake Speech](https://arxiv.org/abs/2604.02913)

The encoder is WavLM Base Plus with the same pinned assets as the local v13
replay. Its model card links to CC BY-SA 3.0 for the weights. The referenced
code license is separate. Only the small newly fitted heads were trained here.
[Pinned publisher card](https://huggingface.co/microsoft/wavlm-base-plus/blob/4c66d4806a428f2e922ccfa1a962776e232d487b/README.md)
· [Weight terms and asset digests](audio-wavlm-v13-protocol.md)

A second readout uses those same heads and compares each candidate range
against speech in two flanking intervals. It subtracts the larger flank mean
and retains the learned endpoint contribution. Each 300 ms flank is separated
from the candidate by 100 ms and needs at least half detected speech. This
readout was designed after inspecting the absolute-score results on the
primary panel, and its comparison on that panel is exploratory. Both its
operating points were fitted on the original genuine calibration readers.

## Noisy held comparison

In each direction, six readers provide fit data, two other readers provide
ordinary calibration clips, and three further readers provide assessment
clips with a different generator. The same held readers appear in the two
directions. Matched original speech, genuine splices, resampled genuine speech
and isolated background measure false alerts. The ICSI mixtures use the same
background waveform in each matched set and an overall −5 dB foreground
level. The exact label and background stems enter evaluation only. The source
bank was previously used in the log-Mel study. [Source terms and split](audio-short-edit-detector-training-2026-11-14.md#source-and-fitting-roles)

Each cell shows useful locations out of 24 staged edits, flagged ordinary
foreground files out of 72, and flagged isolated backgrounds out of 24. A
useful location covers at least half the true edit and reaches 0.30 temporal
intersection over union. Every noisy file had enough speech for at least one
eligible proposal in every compared arm.

| Check | Held Chatterbox useful / ordinary / background | Held MeloTTS useful / ordinary / background |
| --- | ---: | ---: |
| Current short review | 2 / 32 / 0 | 2 / 32 / 0 |
| Log-Mel, strict | 4 / 48 / 12 | 0 / 0 / 0 |
| Log-Mel, wider | 4 / 48 / 12 | 0 / 15 / 2 |
| Frozen encoder, strict | 0 / 0 / 0 | 1 / 2 / 2 |
| Frozen encoder, wider | 1 / 16 / 2 | 1 / 4 / 2 |
| Frozen encoder with flanks, strict | 0 / 0 / 0 | 0 / 0 / 0 |
| Frozen encoder with flanks, wider | 1 / 1 / 0 | 0 / 0 / 0 |

Before applying either cutoff, the frozen encoder put **18/48** noisy edits
above all three of their matched genuine recordings, versus **7/48** for
log-Mel. In dry clips the same comparison was **37/48** versus **6/48**.
The wider flank readout added one useful Chatterbox location to the current
review without adding a genuine or isolated-background alert. It selected
**3.30–4.30 s** against the known **3.30–3.98 s** edit. Its single ordinary
alert was already flagged by the current review. The separate useful MeloTTS
location from the absolute-score head also flagged at the same time in a
matched control. The model's improvements in relative scores therefore still
leave most short edits unlocalized at the tested operating points.

## Separate transfer test

The separate −5 dB test uses 48 retained recordings from four different AMI
meeting rooms. The sources contain three clone edits, four ElevenLabs v4
edits and four Kokoro edits, alongside 33 ordinary controls and four isolated
backgrounds. The test checks exact PCM commitments and excludes any group
with a LibriSpeech host or donor who appears in the primary source bank. One
four-file clone group was excluded. These recordings were used in a previous
project evaluation. ElevenLabs and Kokoro were outside both fit and operating
point roles. The already-fitted points carry through unchanged. The current
app check was rerun on these exact PCM inputs. [Transfer source roles and terms](audio-remote-neighbor-replication-2026-11-12.md#held-sources-and-frozen-comparison)

| Check at the wider point | Useful clone / ElevenLabs / Kokoro | Ordinary flags | Isolated background flags |
| --- | ---: | ---: | ---: |
| Current short review | 0 / 0 / 0 | 5/33 | 0/4 |
| Log-Mel trained on Chatterbox | 0 / 0 / 0 | 6/33 | 0/4 |
| Log-Mel trained on MeloTTS | 0 / 0 / 0 | 18/33 | 0/4 |
| Frozen encoder trained on Chatterbox | 0 / 1 / 0 | 0/33 | 0/4 |
| Frozen encoder trained on MeloTTS | 0 / 1 / 0 | 3/33 | 0/4 |
| Frozen encoder with flanks, either fit | 0 / 0 / 0 | 0/33 | 0/4 |

Neither strict operating point found a useful transfer location. Both wider
encoder fits located the same ElevenLabs edit at **3.44–4.24 s**. The
Chatterbox-trained head selected **3.40–3.90 s**, and the MeloTTS-trained
head selected **3.70–4.20 s**. Every recording was assessed in every arm.
The flanking comparison's initial localization gain did not repeat here.
The candidates remain in evaluation. The different false-alert patterns and
the single added transfer location give a limited basis for changing the
customer-facing detector or its confidence calibration.

## Independent replay

The numeric receipts contain response scores, sample ranges, source digests,
cutoffs and weight-file hashes. Raw speech and frame features stay local. An
offline numeric replay recounts locations, matched-control collisions,
ordinary flags and unassessed files from the supplied data. Native replay also
verifies exact PCM and feature commitments and reconstructs the fitted heads.
The second local CPU run reproduced the feature hashes for all 1,120 distinct
waveforms, the fitted head hashes, the cutoffs and both primary readouts.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
baseline=docs/evaluation-panels/audio-short-edit-detector-training-2026-11-14.json
panels=docs/evaluation-panels
python3 -B -S -m scripts.audio_short_edit_ssl_transfer review "$baseline" \
  "$panels/audio-short-edit-speech-layers-2026-11-14.json" \
  --flank "$panels/audio-short-edit-speech-flanks-2026-11-14.json"
python3 -B -S -m scripts.audio_short_edit_ssl_transfer review-transfer \
  "$panels/audio-short-edit-speech-layer-transfer-2026-11-14.json"
```

Follow the [original source preparation](audio-short-edit-detector-training-2026-11-14.md#independent-replay)
and [separate transfer reconstruction](audio-remote-neighbor-replication-2026-11-12.md#replay)
to recover the bound inputs. Use the pinned speech encoder assets, the same
speech activity and incumbent weights, and the original log-Mel fitted heads.
Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for the local run.

```sh
runner=backend/.venv/bin/python
module=scripts.audio_short_edit_ssl_transfer
"$runner" -B -m "$module" cache "$sources" "$run/cache" "$encoder"
"$runner" -B -m "$module" train "$sources" "$run/cache" "$weights/silero_vad.jit" "$run/layers"
"$runner" -B -m "$module" flank "$sources" "$run/cache" "$weights/silero_vad.jit" "$run/layers" "$run/flank"
"$runner" -B -m "$module" transfer "$sources" "$transfer" "$encoder" "$weights" \
  "$log_mel" "$run/layers" "$run/transfer" --flank "$run/flank"
```
