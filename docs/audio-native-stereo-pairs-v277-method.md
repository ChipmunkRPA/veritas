# Paired native stereo speech construction

Freeze two role-local preparations from the v169c and v172 source and speaker
group commitments and the v274 microphone and window commitments. The
development role has 37 known-origin speech recordings and the holdout role
has 40. Each foreground supplies five stereo files, for 185 development and
200 holdout cases. The four DEMAND sessions have prior channel-one exposure.
This study uses newly qualified ch02 and ch16 views, with washroom and river
in development and kitchen and traffic in holdout. The two roles use
different known source groups, microphone sessions and foreground identities.
[v274 preflight](audio-native-stereo-preflight-v274-result.md)
· [v169c preparation](audio-short-edit-materialization-v169c-result.md)

For each foreground, preserve the exact three-second 16 kHz PCM and inject
the same signal into both recorded background channels. Use the preflight's
exact `200 + 6 × ordinal` second windows, cycling ordinal modulo 16 within
each role and origin family in source manifest order. Preserve sample
alignment. For each tape, create an equal-gain mixture with one shared noise
scale that targets a −5 dB joint stereo signal-to-noise ratio. Keep the
recorded balance between channels. Separately create a forced-asymmetry
mixture targeting +5 dB on the quieter channel and −10 dB on the other,
alternating the quieter channel by the frozen foreground ordinal. These are
controlled foreground and background mixtures. Naturally recorded stereo
speech remains a separate validation question.

For all five views of the same foreground, apply one attenuation factor
bounded by the maximum absolute sum of both channels' speech and noise
components. Quantize the shared clean foreground and each noise residual
separately to signed 16-bit values, add in wider integer precision, and
require exact sample subtraction and the declared clipping headroom.
Recompute SNR from the saved integer components and require the applicable
joint or per-channel targets within 0.1 dB. Save one gain-matched dual-mono
clean file and the four stereo variants.

Reopen every native WAV and verify its exact interleaved bytes. Decode it
through the current production entry point and require the explicit FFmpeg
resampler to produce byte-identical mono PCM. Independently extract both
channels through FFmpeg and compare them with the exact integer construction
and shared zero-origin 48,000-sample clock. Save the private native WAVs and
their production, channel and clock commitments. The worker preserves every
planned case and any unavailable or failed reasons. The separate readout
reconstructs all cases and checks saved native bytes, channel hashes, clock
records and complete accounting. No model execution or calibration is needed.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_pairs_v277 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_pairs_v277 run development
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_pairs_v277 readout development
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_pairs_v277 seal-development
```

The local freeze reads only source metadata and public commitments. The
development worker and readout open development media. The operator can
admit holdout construction only after reviewing and sealing a complete
development replay with all 185 cases prepared. Holdout then has its own
worker and readout under the same frozen procedure.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_pairs_v277 run holdout
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_native_stereo_pairs_v277 readout holdout
```

The existing v108 data-only monitor covers both workers. Private receipts
and media remain local. The aggregate output contains role and case counts
and artifact commitments. Speech authenticity, channel quality selection,
serving evidence, and overall confidence require a separate prospective
comparison.
