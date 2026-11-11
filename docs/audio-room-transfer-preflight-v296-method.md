# Audio room-transfer preflight

The authored preflight verifies a three-second stereo renderer and the production
FFmpeg downmix. It measures no detector performance. A future room-transfer study
would combine known-origin mono speech and separate background audio using
measured stereo room impulse responses. The rendered audio is a simulation of
the acoustic scene. An independent native stereo capture study is still needed.

## Authored checks

Give each impulse response a receiver pair, source position, native sample rate,
linear amplitude unit and two integer arrival offsets on its native clock. An
offset of zero means the delay is already in the coefficients. Preserve both
the embedded and declared timing without independent channel alignment. A source
with only fractional or ambiguous external delay requires a separate intake
procedure. Render the foreground and the background from distinct source
positions measured with the same receiver pair. The mono inputs must each be
exactly 48,000 samples at 16 kHz. Resample each mono input to the impulse
response clock, perform zero-history convolution, and resample both rendered
channels together back to 16 kHz. Use the half-open interval `[0, 48000)` on
the output clock. The filter uses zero padding and the rendered window excludes
the tail after three seconds. The SHA-256 bindings identify each source PCM,
native impulse pair, quantized foreground, quantized residual, stereo WAV and
production mono PCM separately.

Keep the foreground and its room position identical across matched clean,
left-background and right-background views. Set each noisy view to -10 dB joint
SNR using one scalar for the entire two-channel background. Apply one common
clipping-safe scale to both channels of all three views. Retain each channel's
SNR as measured. The authored impulses exercise different signed reflection
coefficients and arrival times, mirrored noise geometry, exact subtraction of
the quantized clean component, and the existing v274/v277 WAV readback, exact
FFmpeg mono PCM and continuous source clock checks. Mirroring the authored
background changes stereo bytes and preserves the mono bytes. The regression tests also
check relative arrivals through the 48 kHz resampling path.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=backend \
  backend/.venv/bin/python -B -m scripts.audio_room_transfer_preflight_v296
```

The command creates and removes temporary authored WAVs and prints a JSON
receipt. The receipt records implementation hashes, the FFmpeg and FFprobe
binary hashes and the numerical library versions. No measured response, external
audio, model weight or existing evaluation media is opened.

Both local readouts reproduced the same stereo and mono byte commitments. Moving
the authored background from left to right reversed the cleaner channel. The
per-channel SNR went from −11.63 and +1.16 dB to +4.22 and −14.69 dB, with
joint SNR about −10 dB for both views. [Authored receipt](evaluation-panels/audio-room-transfer-preflight-v296-authored.json)

## Prospective measured-room study

Use the University of Surrey IoSR room for development. Its 13.6 MB
`angle_0.0.mat` member contains one documented binaural receiver orientation
with 24 separately measured loudspeaker positions. For holdout use one
two-channel stage-audience pair measured in St Paul's Hall by the University
of Huddersfield. Both impulse response releases declare CC BY 4.0. Before
decoding the sources, select the holdout pair from the publisher geometry and
bind the exact source files, channel order, sample rate, linear units, sample
layout and delay conventions. In each room choose distinct horizontal
full-range positions nearest frontal 0°, left 90° and right 90° respectively.
Resolve equal distance from the target by ascending publisher position ID.
Stop room qualification if the source documentation cannot support a verified
pair and the three distinct positions. The authored stage does not pin or
verify the bytes of either measured library. [Surrey release](https://openresearch.surrey.ac.uk/esploro/outputs/dataset/IoSR-Listening-Room-Multichannel-BRIR-Dataset/99512517702346) [Surrey member](https://github.com/IoSR-Surrey/IoSR_ListeningRoom_BRIRs/blob/master/IoSR_ListeningRoom_BRIRs_MATLAB/angle_0.0.mat) [Huddersfield release](https://zenodo.org/records/2639677)

For a later speech experiment, take up to three distinct Hi-Fi TTS speaker
groups per role and four different utterances per group, each paired with the
same relative path in natural, VITS and FastPitch-HiFiGAN directories from the
ODSS publisher archive. That yields 36 source files and 108 room-rendered
cases per role across the three conditions. A local archive check verified
3,778 exact relative-path triplets across eight publisher speaker groups.
The check read seven bounded ZIP metadata ranges, including the catalog,
embedded CC BY-SA 4.0 notice and speaker inventory. It read no recordings.
[ODSS inventory receipt](evaluation-panels/audio-odss-inventory-v297.json)

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_odss_inventory_v297
```

Verify actual PCM, origin, rights, speaker and text
independence before freezing the case roster. Keep complete connected speaker,
prompt and recording groups in one role and exclude earlier exposed source
groups. Require independently qualified background recordings, disjoint by
parent tape between roles, and use the same excerpt for all origins of a paired
prompt in each position. Preserve unusable or unassessed planned rows. A new
offset from a previously used tape is a reused parent. Report each generator and
its matched human-origin speech separately. [ODSS dataset and license](https://zenodo.org/records/8370669) [Hi-Fi TTS source paper](https://www.isca-archive.org/interspeech_2021/bakhturina21_interspeech.pdf)

Before scoring, verify that background placement changes the per-channel SNR
ordering without channel-specific gain, and report the number and spread of
those measured differences for all planned rows. Qualify the selection behavior
with the existing frozen acoustic rule and no authenticity scores. Then use
the unchanged production mono path and fixed AI-only rescue policy to count
correct synthetic calls, new false AI calls, wrong original calls, inconclusive
and unassessed results. Keep media, derived PCM and source identities private
and publish aggregate results and attribution. Confirm the underlying voice and
recording terms at intake, including the ODSS share-alike conditions for any
future release of derived media.
