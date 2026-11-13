# Locate a timing change in one stereo channel

In **Compare with an earlier audio recording**, Chipmunk Verify can find the same audio excerpt at different positions on the left and right channels. The browser and PDF show the channel, signed offset, changed interval and two waveforms. The JSON adds playable clips, decoded sample ranges, source timestamps and checksums.

The check applies to matching continuous native 16 kHz PCM16 stereo WAV or FLAC copies of 1.5–12 seconds. It tests each channel in its recorded order. It requires an exact match at a unique offset, distinctive audio, and unchanged samples on both sides of a short changed interval. It can report separate offsets if both channels move. Ordinary copying or editing can cause this evidence. The measurement carries zero AI-confidence weight.

## Local validation

A development panel used eight retained Surrey source groups. A second panel used twelve previously evaluated source groups in a measured Huddersfield room. The exact channel check found **48 of 48** one-channel moves on the second panel. A correlation check on the mono mix found two. Separate channel correlations with an exact excerpt requirement found 18. The exact channel check also recovered all 72 individual channel moves on the common and opposite offset pairs and reported no findings on 48 gain, channel-swap, whole-channel delay or unchanged control comparisons. The two panels used controlled 200 ms moves.

A separate native challenge used 12 Lombard speaker groups with sources outside those earlier stereo roles. Each scene mixes a plain speaker with an independently recorded Lombard speaker, convolved with published stereo room responses, with competing speech 5 dB louder on the mono mix. The test uses 48 local moves across both channels and directions with signed lags from about 130 to 212 ms. Some moves have nonmatching fade edges. It also uses 60 controls with unchanged or isolated background, local gain, replacement with only the background signal, and repeated audio with ambiguous placement. The implemented check reproduced **48 of 48** signed moves and flagged **zero of 60** controls. Each recording supplies several correlated views. These results cover exact moved excerpts, including moves of the mixture itself. [Publisher speech and terms](https://zenodo.org/records/3736465) · [Published room measurements](https://zenodo.org/records/2639677) · [Challenge replay](evaluation-panels/stereo-channel-timing-challenge-2026-11-12.json)

Chrome showed two waveform and playback controls for the held left-channel move. It found the matching sound 130.125 ms later. The earlier excerpt covers samples [15791, 22144) and the analyzed excerpt covers [17873, 24226). The changed interval is [17793, 24306), including both fades. The API and downloaded JSON and PDF agreed. The independent stereo verifier decoded both original files and confirmed the source coordinates, waveform bins and both WAV excerpts. The comparison created no additional analysis, checkout or job.

The standalone check cannot determine the identity or authenticity of a voice. A separate acoustic-only prototype used annotated positive and background reference intervals to choose an output from a blind stereo separator. It selected the closest candidate on all eight previously tested off-center scenes, with median gains of 5.50 dB overall and 4.89 dB within the edit. It also picked a candidate on three of four background-only controls and remains outside the app. [Earlier blind routing and acoustic sources](audio-blind-huddersfield-transfer-2026-11-12.md)

## Replay

Run the independent source check on both original recordings and the browser JSON.

```sh
python3 -B -IS backend/scripts/verify_stereo_reference.py earlier.wav analyzed.wav comparison.json
```

To reconstruct the licensed local challenge, use the source intake from the short lip timing study and the pinned Huddersfield room archive. Source media stay local.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_stereo_channel_timing_challenge \
  "$sources" "$room_archive" docs/evaluation-panels/stereo-channel-timing-challenge-2026-11-12.json --verify
```
