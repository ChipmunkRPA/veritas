# Engineering preflight of learned genuine-speech reconstruction

A fixed enrollment-conditioned extractor improved the measured foreground-to-error
ratio in all four held noisy deliveries and retained the genuine foreground within
the frozen acoustic limit. Those deliveries represent **two reader/meeting
lineages**, each repeated at two noise levels. This is a small engineering result
with possible model-training corpus overlap. Synthetic speech preservation and
deepfake detection remain unmeasured. The app's assessment stays unchanged.

This follows the [room-point confirmation](audio-room-point12-confirmation-2026-11-19.md),
where lowering the speech point added ordinary alerts under background speech.
The present test asks whether a separate genuine reference can support an
alternative acoustic representation. It supplies no standalone-upload result.

## Inputs and frozen method

Use four retained LibriSpeech target readers, with two development and two held
roles. Each continuous twelve-second host has a separate four-second genuine
reference from a different chapter of the same declared reader. The wrong reference
comes from the other reader in the same role. Target identities have zero declared
overlap across roles.

Four retained AMI meeting groups supply one background per target reader. The two
held meetings are separate from development, with sixteen distinct annotated
participants across the four groups. Annotation-qualified twelve-second excerpts
avoid the retained cuts used by the preceding enrollment preparations. Each fixed
960 ms acoustic control interval has at least 400 ms of annotated overlapping
background voices. This qualification measures annotation coverage, with no voice
attribution to an individual in the mixture.

Each host has clean, −5 dB and −10 dB deliveries. SNR is set on the fixed control
interval, and all three share one peak-safe foreground gain. Round foreground and
noise components separately, add them with exact integer arithmetic, and retain
each scaled background alone. The panel contains twelve genuine deliveries and
eight background-only deliveries, all native 16 kHz mono PCM16 WAV with 192,000
samples. Noise copies share their original reader and meeting lineage.

The exact causal BSRNN checkpoint and its shipped configuration stay fixed. It
uses an embedded ECAPA enrollment encoder. Loading permits tensor-only state and
matches all 948 model state entries. The active path retains the published
configuration, seeded feature dither and original sample clock. It uses two CPU
threads, raw output amplitude and the full supplied reference. It applies no output
peak normalization, enrollment VAD or trimming, truth-based alignment or output
rescaling.

At the fixed original interval, fit each output against the exact known foreground
and noise. Foreground gain gives acoustic retention in decibels. SNR counts the
remaining noise projection and unexplained output energy as error. The unprocessed
mixture supplies the acoustic baseline. Whole-clip measures are retained privately.
Clean inputs have zero added noise, so their floored SNR change has no useful
interpretation. Background-only projection measures accidental resemblance and
supports no claim that the enrolled speaker is present.

The frozen engineering gate requires at least 0.5 dB SNR improvement and at least
−3 dB foreground retention in every held noisy cell. True enrollment must also
produce better SNR than wrong enrollment in at least two-thirds of those cells.
This gate qualifies further acoustic research. It is separate from an AI
confidence score or a customer screening rule.

## Closed results

| Held condition | Genuine deliveries | SNR gain at least 0.5 dB | Foreground retention at least −3 dB | True reference beats wrong reference |
| --- | ---: | ---: | ---: | ---: |
| −5 dB overlapping speech | 2 | 2 | 2 | 1 |
| −10 dB overlapping speech | 2 | 2 | 2 | 2 |
| Total, two underlying lineages | 4 | 4 | 4 | 3 |

All three frozen engineering requirements pass. Across the four held cells, two
SNR gains fall in the interval [5, 10) dB and two in [10, 15) dB. Three foreground
gains lie in [−3, 0) dB and one in [0, 3) dB. All four clean true-reference outputs
have foreground gains in [0, 3) dB. The public receipt uses these broad bands and
group counts to limit disclosure of individual acoustic responses.

One held noisy delivery has equal or better SNR with the wrong reference. The
reference therefore provides inconsistent selectivity even in this small panel.
The acoustic success establishes neither speaker identity nor reliable exclusion
of interfering voices. A future short-speech detector would still need matched
ordinary controls and direct synthetic-retention checks on its proposed input.

## Source exposure and licensing

The genuine originals are from **LibriSpeech train-clean-100**. The shipped model
configuration names `data/clean/train-100` training inputs. Exact upstream training
utterance and reader membership remain unresolved. This same-corpus overlap can
favor reconstruction. The held role is separate within this project and supplies
no unseen-reader or model-training-disjoint validation.

The genuine recordings, enrollment sources and four AMI meeting groups have prior
project exposure. New background excerpts and strict current-role separation
preserve the engineering comparison while keeping that reuse visible. The four
held noise-level deliveries remain correlated copies of two underlying lineages.

The [REAL-TSE publisher](https://github.com/REAL-TSE/wesep-real-tse/blob/d9ed77d788e9bc8ca08d15854defe404e4593773/README.md)
grants Apache-2.0 terms for its inference code and linked checkpoints. The embedded
[WeSpeaker component](https://github.com/wenet-e2e/wespeaker/blob/8f53b6485d9f88a207bd17e7f8dba899495ec794/docs/pretrained.md)
uses Apache-2.0 code and the publisher's stated CC BY 4.0 terms for VoxCeleb
pretrained weights. The selected path uses the
[TorchAudio Kaldi frontend](https://github.com/pytorch/audio/blob/v2.7.1/LICENSE)
under BSD-2-Clause. The [LibriSpeech recordings](https://www.openslr.org/12/) and
[AMI recordings and annotations](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
are released under CC BY 4.0.

The optional USEF branch has separate noncommercial terms. Its source was removed
before model imports, the disabled branch uses a fail-closed stub, and the selected
checkpoint has no USEF tensor entries. That optional component contributes no
execution or result here. The published receipt identifies the exact global
checkpoint and configuration digests without distributing their weights.

## Independent checks and public replay

A separate implementation verified eight original FLACs and three whole Parquet
pins, different-chapter enrollment, the four original AMI cuts and participant
metadata. It replayed all twenty delivered WAVs through FFmpeg, checked 3,840,000
input sample values and 1,536,000 additive component sample identities, and passed
seven authored acoustic-grader controls.

The source run closed with forty raw outputs plus one exact repeat in about two
minutes. Two authored baseline calls separately checked safe runtime operation
before source inference. Independent original-clock arithmetic recomputed 480
scalar fields from all 7,680,000 saved output samples and matched every frozen
gate. A separate follow-up tested clone retention on the same reference setup.
[Synthetic speech retention](audio-learned-enrollment-conditional-retention-2026-11-19.md)

The [anonymous scalar receipt](evaluation-panels/audio-learned-enrollment-genuine-preflight-2026-11-19.json)
contains grouped decisions, coarse gain bands, global asset digests and source
citations. Reader and meeting identities, source positions, individual waveform
hashes, private paths, per-case response vectors, model features and audio remain
private. The public reviewer checks the pinned receipt's arithmetic with Python's
standard library and rejects twelve authored corruptions.

```sh
python3 -I -B -S backend/scripts/review_audio_learned_enrollment_genuine_preflight.py \
  docs/evaluation-panels/audio-learned-enrollment-genuine-preflight-2026-11-19.json \
  --self-test
```

This public recount covers the exported arithmetic. Exact original-source and
model replay requires retained private custody. Deepfake sensitivity, false alarms,
synthetic preservation, speaker attribution and deployment remain outside the
measured result. Overall confidence and all production detection settings stay
unchanged.
